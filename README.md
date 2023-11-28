# Compact Transformers for Small-Scale Learning: A Reproducibility Study
This repository hosts the code for our reproduction [Compact Transformers for Small-Scale Learning: A
Reproducibility Study](https://github.com/MLReproHub/CCT/blob/3ee791470b707ffa7e8fb3b0ad76d4bc3d3bf756/Compact%20Transformers%20for%20Small-Scale%20Learning%20-%20Charisoudis%2C%20A.%2C%20Ekman%20von%20Huth%2C%20S.%2C%20Jansson%2C%20E.pdf) of the paper ["Escaping the Big Data Paradigm with Compact Transformers"](https://arxiv.org/abs/2104.05704) by Hassani et al.


### Creating a conda environment: 
`$conda env create -f environment.yml`

### Activating the conda environment: 
`$conda activate`

### Updating the shared dependencies: 
`$conda env export > environment.yml`<br/>
The last prefix line can be omitted as it doesn't make any difference.
