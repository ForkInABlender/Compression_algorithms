# Compression_algorithms
To compress and uncompress data even for LLM dot products

# What makes these unique:

What makes them unique is the way they compress information losslessly.

the ctypes based compactor allows for full object and running object state compact and restore, either to and from memory or disk, and back.

The numpy compactor uses a unique binarial format called "F642C" for storing and utilizing larger than normal numpy arrays. It chunks/slices and is useful devices like Android where large numerical representation is difficult or blows out the default memory table.
