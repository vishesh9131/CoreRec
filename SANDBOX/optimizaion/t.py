import argparse
from tasty import convert_python_to_cpp

def main():
    parser = argparse.ArgumentParser(description='Convert Python code to C++ code.')
    parser.add_argument('input_file', help='The Python file to convert.')
    parser.add_argument('output_file', help='The output C++ file.')

    args = parser.parse_args()

    try:
        with open(args.input_file, 'r') as infile:
            python_code = infile.read()
    except IOError:
        print(f"Error: Unable to read file {args.input_file}")
        return

    cpp_code = convert_python_to_cpp(python_code)

    try:
        with open(args.output_file, 'w') as outfile:
            outfile.write(cpp_code)
    except IOError:
        print(f"Error: Unable to write to file {args.output_file}")
        return

    print(f"Conversion complete. C++ code written to {args.output_file}")

if __name__ == "__main__":
    main()
