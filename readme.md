# GNC Simulation - dev branch

Active Branch for simulation development

> [!WARNING]
Do Not push code to main. Use PULL REQUESTS between dev and main


## Setting Up
### First time
1. Run `python -m venv .venv` to create a virtual environment (venv)
2. Run `source .venv/bin/activate` to activate the venv
3. Run `pip install -r requirements.txt`

### Each time
2. Run `source .venv/bin/activate` to activate the venv
3. Once done wit hdevlopment, run `deactivate` to exit the venv

## Style Guide
> [!IMPORTANT]
> PEP8 formatting rules

1. Use <u> ruff </u> to auto-check formatting
``` pip install ruff ``` or use VSCode Extension

2. Before any function/class definition, provide the following details in a multi-line comment
```
    '''
        FUNCTION <function name>

        <function purpose>
        
        INPUTS:
            <Numbered list of inputs>
            
        OUTPUTS:
            <numbered list of outputs>
    '''  
```

3. No Loose functions
    - All files will have <u> one and only one </u> class definition within them
    - There will not be any functions not present within a class definition
    - If it does not make sense to have a class, consider merging the function with its caller

## Code Architecture
Refer to the code architecture <a href="https://www.notion.so/Physics-Model-Simulation-Architecture-10648018d82a80d4a90ce8fb38b47777">here</a>


