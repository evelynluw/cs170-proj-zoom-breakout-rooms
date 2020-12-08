import sys
with open('what.txt', 'a+') as f:
    original_stdout = sys.stdout
    sys.stdout = f # Change the standard output to the file we created.
    print('HEY THERE. DO YOU SEE THIS?')
    sys.stdout = original_stdout # Reset the standard output to its original value
print("I'm DONE")