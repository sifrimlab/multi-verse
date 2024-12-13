
# Cobolt: Joint analysis of multimodal single-cell sequencing data 

## Instructions to run the Cobolt Dockerfile:

After creating the image (here referred to as image_cobolt) and creating a volume (called volume_cobolt)

1. Check that the data are correctly accessible with:

```bash
docker run --rm -it -v volume_cobolt:/data image_cobolt:latest sh -c "ls /data"
```

2. If data are scuccesfully accessible run the docker:

```bash
docker run --rm -v volume_cobolt:/CoboltModel/data image_cobolt:latest
```

## License
[COBOLT](https://github.com/epurdom/cobolt)