# Data generator

To generate new social media data, use `generate.py` script.

There will be generated two files:

- output - files stores generated samples
- analysis.json - stores data to be compared with data returned by notebooks

## Arguments:

```
-u, --users [number]         number of users, which will be "using" the social media
-s, --samples [number]       number of samples to generate
-o, --offest [number]        max time between each sample
output                       file path, where results will be stored
```