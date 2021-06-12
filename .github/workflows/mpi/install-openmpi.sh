#!/bin/bash
set -e
case `uname` in
Linux) set -x;
  sudo apt update
  sudo apt install -y openmpi-bin libopenmpi-dev
  ;;
Darwin) set -x;
  brew install openmpi
  ;;
esac
