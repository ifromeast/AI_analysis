
creat environment at all node by
```
conda env create -f environment.yml
```

start ray at head node by
```
ray start --head --dashboard-host=0.0.0.0
```

start ray at worker node by
```
ray start --address='10.157.150.10:6379'
```

