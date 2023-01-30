import argparse
from chat_processor import process_chat_data, test_process_chat_data

def main():
    parser = argparse.ArgumentParser(description='Process chat data.')
    parser.add_argument('csv', type=str, help='The path to the CSV file containing the chat data.')
    args = parser.parse_args()
    test_process_chat_data(args.csv)
    process_chat_data(args.csv)

if __name__ == '__main__':
    main()
