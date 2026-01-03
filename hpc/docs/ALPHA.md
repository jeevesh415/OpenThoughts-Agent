# Setting alpha for evals

Already set up `/data/horse/ws/ryma833h-DCFT_Shared/`

Alpha only has access to `horse` not `cat`. 

Since they share the same home, we need to do the following in the bashrc

```
if [[ "$(hostname)" == *"alpha.hpc.tu-dresden.de"* ]]; then
        export DCFT=/data/horse/ws/ryma833h-DCFT_Shared
        source $DCFT/dcft_private/hpc/dotenv/alpha.env
else
        export DCFT=/data/cat/ws/ryma833h-dcft
        source $DCFT/dcft_private/hpc/dotenv/zih.env
fi
source $DCFT_PRIVATE/hpc/scripts/common.sh
```
Had to delete the symlink for ~/.cache/huggingface because I was getting `RuntimeError: Data processing error: I/O error: File exists (os error 17)` 
Also the symlink was going to `cat`, which alpha doesn't have access to 
