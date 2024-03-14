import os.path
import urllib.request
from urllib.error import HTTPError


def download_config_and_weights_with_mim(config, destination_directory, mm_package="mmdet", print_prefix="DOCKER"):
    from mim.commands.download import _download_configs
    from mim.commands.search import get_model_info
    try:
        config_file = os.path.join(destination_directory, config + ".py")
        checkpoint_file = os.path.join(destination_directory,
                                       _download_configs(mm_package, [config], destination_directory)[
                                           0])
        return config_file, checkpoint_file
    except ValueError as e:
        print(f"[{print_prefix}] Your config is not available")
        print(f"[{print_prefix}] Here is a list of available configs:")
        print("\n".join([f"[{print_prefix}] " + v for v in sorted(get_model_info(mm_package, to_dict=True).keys())]))
        print(f"[{print_prefix}] Your config was:", config)
        return None, None


def download_weights(url, destination_directory, print_prefix="DOCKER"):
    output_file_path = os.path.join(destination_directory, url.split("/")[-1])
    if not os.path.exists(output_file_path):
        print(f"[{print_prefix}] Downloading weights from:", url, "to", output_file_path)
        try:
            urllib.request.urlretrieve(url, output_file_path)
        except HTTPError as e:
            print(f"[{print_prefix}] Unable to download {url} ({e})")
            return None
    else:
        print(f"[{print_prefix}] Found weights at:", output_file_path)
    return output_file_path
