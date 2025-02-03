template_text = open("template.txt", 'r').read()
words = template_text.split()

for i in range(len(words)):
    modified_words = words[:]
    modified_words[i] = modified_words[i].upper()
    file_content = " ".join(modified_words)
    
    with open(f"file_{i+1}.txt", "w") as f:
        f.write(file_content)
    
    print(f"Generated file_{i+1}.txt with content: {file_content}")