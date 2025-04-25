import json
import sys

def combineFiles(file_paths, output_file):
    combinedData = []
    
    for file_path in file_paths:
        
         with open(file_path, 'r') as f:

            data = json.load(f)

            combinedData.extend(data)

    with open(output_file, 'w') as outfile:

        json.dump(combinedData, outfile)


file_paths = sys.argv[1:-1]
output_file = sys.argv[-1]

print(file_paths)
print(output_file)

combineFiles(file_paths, output_file)
