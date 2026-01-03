# TACC Setup

## Access Instructions

First, make an account on the [TACC portal](https://accounts.tacc.utexas.edu/begin).

After your account is approved, ping Negin on Discord (`neginraoof`) so that you can be added to the allocation (this might take some time because it requires approval by TACC admins).

## Helpful Resources

Official Documentation - https://docs.tacc.utexas.edu/hpc/vista/
George's Common Pitfalls - https://docs.google.com/document/d/1URcWe8mLQF8HMNre7vZwgk6TiRNxFxVBwK_HCcRXQdM/edit?usp=sharing

## Shared Working Directory

The following path is where all group readable and group editable files are stored. This will contain the "golden" / "production" versions of the `dc-agent`, `skyRL` and `terminal-bench` installations. Please make non-critical path breaking changes on your own installation so others don't get blocked. 
```
/scratch/08134/negin/dc-agent-shared/
```

Everyone in the `G-827553` group will have access to this. This is the [`DataComp` projec](https://tacc.utexas.edu/portal/projects/66366) (CCR24067) on TACC. 

The [official setup instructions](https://docs.tacc.utexas.edu/tutorials/sharingprojectfiles/) for shared working directories is how we set this directory up initially. (You don't need to run this again). 
```bash
chmod g+X $WORK/..
chmod g+X $WORK
cd $WORK
mkdir dc-agent-shared
chgrp -R G-827553 dc-agent-shared
chgrp G-827553 $WORK
chgrp G-827553 $WORK/..
chmod g+s dc-agent-shared
newgrp G-827553
chmod g+rwX dc-agent-shared
```

Put the following in your `~/.bashrc`: 
```bash
##########
# Umask
#
# If you are in a group that wishes to share files you can use
# "umask". to make your files be group readable.  Placing umask here
# is the only reliable place for bash and will insure that it is set
# in all types of bash shells.
umask 007
```
