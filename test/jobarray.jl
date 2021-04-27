using GCMAES, MPI

ENV["SLURM_ARRAY_TASK_ID"] = 1
ENV["SLURM_ARRAY_TASK_COUNT"] = 2

@mpirun GCMAES.make_virtual_jobarray(2)

rank = GCMAES.myrank(MPI.COMM_WORLD)
@show rank, GCMAES.myrank(), GCMAES.partjob(1:8), GCMAES.part(GCMAES.partjob(1:8))
