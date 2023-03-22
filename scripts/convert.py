import json

# Merge Unnatural Instructions dataset with the Alpaca dataset

with open("../dataset/raw_alpaca_data.json") as f:
    content = f.read()

alpaca_data = json.loads(content)
print("Read " + str(len(alpaca_data)) + " prompts from raw_alpaca_data.json")

with open("../dataset/unnatural_instructions.jsonl") as f:
    content = f.read()

# parse jsonl file
json_data = [line for line in content.splitlines()]

# iterate over the list
for i in json_data:

    try:
        jsonline = json.loads(i)
    except:
        print("Failed to parse: " + i)
        continue

    for instance in jsonline['instances']:
        formatted_instruction = instance["instruction_with_input"].replace(instance['input'], "").strip().strip("\n")

        newPrompt = {
            "instruction": formatted_instruction,
            "input": instance["input"],
            "output": instance["output"]
        }

        alpaca_data.append(newPrompt)

    if 'reformulations' not in jsonline:
        continue
    for instance in jsonline['reformulations']:
        formatted_instruction = instance["instruction_with_input"].replace(instance['input'], "").strip().strip("\n")
        newPrompt = {
            "instruction": formatted_instruction,
            "input": instance["input"],
            "output": instance["output"]
        }

        alpaca_data.append(newPrompt)

# serialize alpaca_data to json and write out to file
with open('../dataset/training_data.json', 'w') as outfile:
    json.dump(alpaca_data, outfile)

print("Wrote " + str(len(alpaca_data)) + " prompts to training_data.json")
