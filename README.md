# BilbyFlow
A fast and efficient package built with Bilby design principles. Detailed documentation can be found at https://bilbyflow.readthedocs.io/en/latest/. 



## Developer Notes

When pushing to the main branch a test suite should now run. 

If you are simply updating documentation then add `[skip ci]` to your commit message somewhere, and it will 
not do the full test suite. Should only take a minute on CPU (literally, ~62s) but still wouldn't be great 
if there's a large number of requests at once.

