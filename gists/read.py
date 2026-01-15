input_file = 'file.input'
config = dict()
with open(input_file, 'r') as f:
    exec(f.read(), {}, config)

print(config) # This will output: my_data_case

# This injects every key in 'config' as a variable in your script
locals().update(config)

# Now you can use them directly:
print(__rloc__)
print(__sample_size__)