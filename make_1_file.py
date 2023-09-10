
import time
CURRENT_TIME = time.strftime("%Y-%m-%d_%H:%M")



save_path = "/home/yoavharlap/work/dates/dates.txt" 



file = open(save_path, "a")
a = file.write('\n' + CURRENT_TIME)
file.close()
print(a)



with open(save_path, 'r') as file:
    # Read the file content
    content = file.read()
    # Split the content into lines and get the last line
    last_line = content.strip().split('\n')[-1]
    # Extract the desired string from the last line
    my_string = last_line.strip()
    print(my_string)
    print(CURRENT_TIME)