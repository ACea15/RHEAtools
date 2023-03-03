#!/usr/bin/zsh
NUM_MODES=5
DIR1=$HOME/programs/RHEAtools
for i in {0..$NUM_MODES}
  do
      echo "Copying Interpolated Mode $i"
      sshpass -f $DIR1/examples/pas ssh ac5015@login.hpc.imperial.ac.uk << EOF
      cd $HOME/runs	     
      mkdir -p "ONERA_fac10/Deformation/M$i"
  exit
EOF
  sshpass -f $DIR1/examples/pas scp $DIR1/data/out/ONERA/SU2_mesh/M$i/sbw_fordef.dat ac5015@login.hpc.imperial.ac.uk:$HOME/runs/ONERA_fac10/Deformation/M$i/sbw_fordef.dat
done
