#!/usr/bin/env python
import argparse, sys

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--config', type=str, help='path to yaml config')
  args = ap.parse_args()
  print(f"[TODO] {__file__} running. Use --config to pass a YAML.")

if __name__ == "__main__":
  sys.exit(main())
