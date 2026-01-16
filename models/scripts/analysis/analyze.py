import argparse

def main(csv_path):
    return



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Analysis of predicted results from ML-Agents.")
    parser.add_argument("--csv-path",dest="csv_path",type=str, required=True)
    args = parser.parse_args()
    main(csv_path=args.csv_path)
