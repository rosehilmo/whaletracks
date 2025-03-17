#! /bin/bash
# Sets up data and environment

/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

bash setup_env.sh
bash setup_data.sh
