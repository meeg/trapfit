# setup
You need to compile the C code:
```
gcc -shared -fPIC -o trapfit.so trapfit.c
```

You also need the DC vs. time CSV files, which I'm not putting here since this is a public repo and in principle this is private data.

# running
Analytic linear fit (fast, seems to work?):
```
./trapfitlin.py dc_TEMP170K-steps4.csv
```

Iterative Minuit fit (slow, doesn't really work):
```
./trapfitminuit.py dc_TEMP170K-steps4.csv
```
