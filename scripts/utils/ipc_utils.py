import os
import socket
import struct
from io import BytesIO

import cv2
import mmap
import numpy as np
import posix_ipc
import time


class IPCCommunication:
    NUMPY_DTYPES = [np.bool_, np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64, None,
                    None, np.float32, np.float64, np.float128, np.complex64, np.complex128, np.complex256, None, None,
                    None, None, None, None, np.float16]

    def __init__(self, server_mode=False, print_prefix=None, client_retry_interval=5, server_retry_interval=2):
        self.server_mode = server_mode
        self.print_prefix = print_prefix
        self.client_retry_interval = client_retry_interval
        self.server_retry_interval = server_retry_interval

        if self.print_prefix is None:
            self.print_prefix = "SERVER" if self.server_mode else "CLIENT"

        self.file = None

    def connect(self, destination):
        if self.server_mode:
            self._connect_server(destination)
        else:
            self._connect_client(destination)

    def _connect_server(self, destination):
        self.print("Run server with:", destination)

    def _connect_client(self, destination):
        self.print("Trying to connect to server with:", destination)

    def start_send(self):
        raise NotImplementedError("")

    def flush(self):
        raise NotImplementedError("")

    def receive(self):
        raise NotImplementedError("")

    def close(self):
        raise NotImplementedError("")

    def print(self, *args):
        print(f"[{self.print_prefix}]", *args)

    def _write_uint64(self, i):
        self.file.write(struct.pack(">Q", i))

    def _read_uint64(self):
        return struct.unpack(">Q", self.file.read(8))[0]

    def _read_bytes(self, num_bytes):
        return self.file.read(num_bytes)

    def _write_bytes(self, b):
        return self.file.write(b)

    def write_with_length(self, callback):
        offset_before = self.file.tell()
        self._write_uint64(0)
        callback()
        offset_after = self.file.tell()

        # rewrite payload size
        self.file.seek(offset_before)
        self._write_uint64(offset_after - offset_before - 8)
        self.file.seek(offset_after)

    def read_with_length(self, callback):
        num_bytes_body = self._read_uint64()
        callback(num_bytes_body)

    def write_string(self, s):
        self.write_with_length(lambda: self._write_bytes(s.encode(encoding='UTF-8')))

    def read_string(self):
        s = ""

        def cb_read_string(num_bytes_body):
            nonlocal s
            b = self.file.read(num_bytes_body)
            s = str(b, encoding='UTF-8')

        self.read_with_length(cb_read_string)
        return s

    def write_numpy_array(self, array):
        def cb_write_data():
            if array.flags['C_CONTIGUOUS']:
                self._write_bytes(array.data)
            else:
                self._write_bytes(array.tobytes())

        def cb_write_shape_tuple():
            for d in array.shape:
                self._write_uint64(d)

        def cb_write_numpy_array():
            # write dtype
            self._write_uint64(array.dtype.num)

            # write shape tuple
            self.write_with_length(cb_write_shape_tuple)

            # write actual data
            self.write_with_length(cb_write_data)

        if array is None:
            array = np.empty(0)
        self.write_with_length(cb_write_numpy_array)

    def read_numpy_array(self):
        dtype = None
        shape = []
        data = None

        def cb_read_data(num_bytes_body):
            nonlocal data
            data = self._read_bytes(num_bytes_body)

        def cb_read_shape_tuple(num_bytes_body):
            nonlocal shape
            offset_before = self.file.tell()
            while (self.file.tell() - offset_before) < num_bytes_body:
                shape.append(self._read_uint64())

        def cb_read_numpy_array(_):
            nonlocal dtype
            # read dtype
            dtype_int = self._read_uint64()
            dtype = self.NUMPY_DTYPES[dtype_int]

            # read shape tuple
            self.read_with_length(cb_read_shape_tuple)

            # read actual data
            self.read_with_length(cb_read_data)

        self.read_with_length(cb_read_numpy_array)
        array = np.ndarray(shape=shape, dtype=dtype, buffer=data)

        if array.shape == (0,):
            array = None

        return array

    def write_opencv_image(self, image, input_encoding, compression="none"):
        # compress image if desired and required
        if image is not None and compression != "none":
            already_compressed = len(image.shape) == 1 and image.dtype == np.uint8
            if not already_compressed:
                start_time_compression = time.time()
                size_before = image.nbytes
                image = np.array(cv2.imencode('.' + compression, image)[1])
                size_after = image.nbytes
                self.print("Duration [compression]:", time.time() - start_time_compression,
                           "--- Compression Ratio:", size_after / size_before)

        def cb_write_opencv_image():
            # write compression string ("none", "jpeg", "png")
            self.write_string(compression)

            # write encoding string
            # it must correspond to left side of "cv::ColorConversionCodes" without "COLOR_" part, e.g. "BGR", "RGB",
            # "Bayer_RGGB", ... (see e.g. https://docs.opencv.org/3.4/d8/d01/group__imgproc__color__conversions.html)
            self.write_string(input_encoding)

            # write corresponding numpy array
            self.write_numpy_array(image)

        self.write_with_length(cb_write_opencv_image)

    def read_opencv_image(self, desired_output_encoding="passthrough"):
        image = np.empty((0,))
        compression = ""
        input_encoding = ""

        def cb_read_opencv_image(_):
            nonlocal image, compression, input_encoding
            # read compression string
            compression = self.read_string()

            # read encoding string
            input_encoding = self.read_string()

            # read actual numpy array
            image = self.read_numpy_array()

        self.read_with_length(cb_read_opencv_image)

        # empty images are also allowed
        if image is None:
            return image

        # print how image was transferred
        self.print("Compression: " + compression + ", Encoding: " + input_encoding + ", Image shape: " +
                   str(image.shape) + ", Image dtype: " + str(image.dtype))

        # decompress image if required
        if compression != "none":
            image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
            input_encoding = "BGR"

        # convert encoding if desired
        if desired_output_encoding != "passthrough" and input_encoding != desired_output_encoding:
            start_time_convert_color = time.time()
            color_conversion_code = getattr(cv2, "COLOR_" + input_encoding + "2" + desired_output_encoding)
            image = cv2.cvtColor(image, color_conversion_code)
            self.print("Duration [convert_color]:", time.time() - start_time_convert_color)

        return image


class TcpSocketCommunication(IPCCommunication):
    def __init__(self, server_mode=False, print_prefix=None, client_retry_interval=5, server_retry_interval=2):
        super().__init__(server_mode, print_prefix, client_retry_interval, server_retry_interval)
        self.server_socket = None
        self.socket = None

    def _connect_server(self, destination):
        super()._connect_server(destination)

        port = destination[0]
        while True:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                self.server_socket.bind(("0.0.0.0", port))
                break
            except OSError:
                self.print(f"Unable to bind to socket. I will try again in {self.server_retry_interval}s.")
                time.sleep(self.server_retry_interval)
        self.server_socket.listen()
        self.socket, address_port_tuple = self.server_socket.accept()
        if self.socket:
            self.print(f"Connected by {address_port_tuple}")

    def _connect_client(self, destination):
        super()._connect_client(destination)

        ip_address, port = destination
        time.sleep(self.client_retry_interval)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        while True:
            try:
                self.socket.connect((ip_address, port))
                self.print("Successfully connected to Docker container")
                break
            except ConnectionRefusedError:
                self.print(f"Unable to connect. I will try again in {self.client_retry_interval}s")
                time.sleep(self.client_retry_interval)

    def start_send(self):
        self.file = BytesIO()
        self._write_uint64(0)  # write dummy size, will be corrected later

    def flush(self):
        # rewrite payload size
        offset_before = self.file.tell()
        self.file.seek(0)
        self._write_uint64(offset_before - 8)

        # send buffer via TCP socket
        self.socket.sendall(self.file.getbuffer())

    def receive(self):
        data = self.socket.recv(8)
        num_bytes_body = struct.unpack(">Q", data)[0]
        payload = bytearray()
        while True:
            bytes_to_receive = num_bytes_body - len(payload)
            if bytes_to_receive == 0:
                break
            payload.extend(self.socket.recv(bytes_to_receive))
        self.file = BytesIO(payload)

    def close(self):
        if self.socket:
            self.socket.close()
        if self.server_socket:
            self.server_socket.close()


class SharedMemoryCommunication(IPCCommunication):
    def __init__(self, server_mode=False, print_prefix=None, client_retry_interval=5, server_retry_interval=2):
        super().__init__(server_mode, print_prefix, client_retry_interval, server_retry_interval)
        self.memory = None
        self.semaphore = None
        self.map_file = None
        self.sequence = 12345
        self.memory_view = None

    def _connect_server(self, destination):
        super()._connect_server(destination)
        shared_memory_name, semaphore_name = destination
        while True:
            path = "/dev/shm" + shared_memory_name
            if os.path.exists(path):
                break
            self.print(f"Unable to connect. I will try again in {self.server_retry_interval}s")
            time.sleep(self.client_retry_interval)
        self.semaphore = posix_ipc.Semaphore(semaphore_name)
        self.memory = posix_ipc.SharedMemory(shared_memory_name)
        self.file = mmap.mmap(self.memory.fd, self.memory.size)
        self.memory.close_fd()
        self.memory_view = memoryview(self.file)
        self.semaphore.release()
        self.receive()
        result = self.read_string()
        if result == "Hello Server":
            self.print("Connection successful")
        else:
            raise RuntimeError("Unexpected initialization sequence:", result)
        self.start_send()
        self.write_string(result + "+ack")
        self.flush()

    def _connect_client(self, destination):
        super()._connect_client(destination)
        shared_memory_name, semaphore_name = destination
        self.semaphore = posix_ipc.Semaphore(semaphore_name, flags=posix_ipc.O_CREAT)
        self.memory = posix_ipc.SharedMemory(shared_memory_name, flags=posix_ipc.O_CREAT, size=1024 * 1024 * 30)
        self.file = mmap.mmap(self.memory.fd, self.memory.size)
        self.memory.close_fd()
        self.memory_view = memoryview(self.file)
        self.semaphore.acquire()

        init_string = "Hello Server"
        self.start_send()
        self.write_string(init_string)
        self.flush()

        self.receive()
        result = self.read_string()
        if result is not None and init_string + "+ack" == result:
            self.print("Connection successful")
        else:
            raise RuntimeError("Unexpected initialization sequence:", result)

    def _read_bytes(self, num_bytes):
        # faster than _read_bytes from superclass (zero copy read)
        offset_before = self.file.tell()
        offset_after = offset_before + num_bytes
        self.file.seek(offset_after)
        return self.memory_view[offset_before:offset_after]

    def _write_bytes(self, b):
        # only slightly faster than _write_bytes from superclass (worth the extra code?)
        offset_before = self.file.tell()
        if type(b) is memoryview:
            bytes_to_write = b.shape[0] * b.strides[0]
            offset_after = offset_before + bytes_to_write
            if b.shape[0] == 0:
                return
            self.memory_view[offset_before:offset_after] = b.cast('B', (bytes_to_write,))
        else:
            bytes_to_write = len(b)
            offset_after = offset_before + bytes_to_write
            self.memory_view[offset_before:offset_after] = b
        self.file.seek(offset_after)

    def start_send(self):
        self.file.seek(0)
        self.file.write(struct.pack(">Q", self.sequence))

    def flush(self):
        self.sequence += 1
        self.semaphore.release()
        time.sleep(0.003)

    def receive(self):
        while True:
            self.semaphore.acquire()
            self.file.seek(0)
            sequence = self._read_uint64()
            if sequence == self.sequence:
                break
            self.semaphore.release()
            self.print("Counterpart did not acquire semaphore fast enough!")
            time.sleep(0.003)
        self.sequence += 1

    def close(self):
        if self.memory_view:
            del self.memory_view
        if self.server_mode:
            self.semaphore.close()
            self.file.close()
        else:
            self.file.close()
            self.memory.unlink()
            self.semaphore.release()
            self.semaphore.unlink()
