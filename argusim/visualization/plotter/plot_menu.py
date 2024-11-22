import tkinter as tk


# inspired by chatGPT
def get_user_selections(options):
    NUM_OPTIONS = len(options)
    # Create the main window
    root = tk.Tk()
    root.title("Select which plots to create")

    # List to keep track of the checkboxes and their variables
    vars = []
    usr_input = []

    # Function to toggle the state of all checkboxes
    def select_all():
        for var in vars:
            var.set(True)

    # Function to toggle the state of all checkboxes
    def clear_all():
        for var in vars:
            var.set(False)

    def get_states():
        for var, option in zip(vars, options):
            if var.get():
                usr_input.append(option)
        root.destroy()

    # Button to check all checkboxes
    toggle_button = tk.Button(root, text="Select All", command=select_all)
    toggle_button.grid(row=0, column=0, columnspan=2)

    # Button to uncheck all checkboxes
    toggle_button = tk.Button(root, text="Clear All", command=clear_all)
    toggle_button.grid(row=1, column=0, columnspan=2)

    # Create several checkboxes
    for i in range(NUM_OPTIONS):
        var = tk.BooleanVar()  # Variable to track the state of the checkbox
        cbox = tk.Checkbutton(root, text=f"{options[i]}", variable=var)
        cbox.grid(row=i + 2, column=0, sticky="w")
        vars.append(var)

    # Button to print the status of checkboxes
    status_btn = tk.Button(root, text="Create Plots", command=get_states)
    status_btn.grid(row=NUM_OPTIONS + 3, column=0, columnspan=2)

    # Run the application
    root.mainloop()
    return usr_input
