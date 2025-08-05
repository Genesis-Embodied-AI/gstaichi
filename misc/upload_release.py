import os
import subprocess
import sys

import requests


def upload_gstaichi_version():
    username = os.getenv("METADATA_USERNAME")
    password = os.getenv("METADATA_PASSWORD")
    url = os.getenv("METADATA_URL")
    for filename in os.listdir("./dist"):
        filename = filename[: len(filename) - 4]
        parts = filename.split("-")
        payload = {"version": parts[1], "platform": parts[4], "python": parts[2]}
        try:
            response = requests.post(
                f"https://{url}/add_version/detail",
                json=payload,
                auth=(username, password),
                timeout=5,
            )
            response.raise_for_status()
        except requests.exceptions.ConnectionError as err:
            print("Updating latest version failed: No internet,", err)
        except requests.exceptions.HTTPError as err:
            print("Updating latest version failed: Server error,", err)
        except requests.exceptions.Timeout as err:
            print("Updating latest version failed: Time out when connecting server,", err)
        except requests.exceptions.RequestException as err:
            print("Updating latest version failed:", err)
        else:
            response = response.json()
            print(response["message"])


def upload_artifact(is_gstaichi):
    pwd_env = "PROD_PWD" if is_gstaichi else "NIGHT_PWD"
    twine_password = os.getenv(pwd_env)
    if not twine_password:
        sys.exit(f"Missing password env var {pwd_env}")
    command = [sys.executable, "-m", "twine", "upload"]
    if not is_gstaichi:
        command.extend(["--repository-url", "https://pypi.gstaichi.graphics/simple/"])
    uname = "__token__" if is_gstaichi else os.getenv("NIGHT_USERNAME")
    command.extend(["--verbose", "-u", uname, "-p", twine_password, "dist/*"])

    try:
        subprocess.check_call(command)
    except subprocess.CalledProcessError as e:
        sys.exit(f"Twine upload returns error {e.returncode}")


if __name__ == "__main__":
    if os.getenv("GITHUB_REPOSITORY", "gstaichi-dev/gstaichi") != "gstaichi-dev/gstaichi":
        print("This script should be run from gstaichi repo")
        sys.exit(0)
    is_gstaichi = os.getenv("PROJECT_NAME", "gstaichi") == "gstaichi"
    upload_artifact(is_gstaichi)
    if is_gstaichi:
        upload_gstaichi_version()
