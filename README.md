# GSP25_particle_SFC



## Getting started (Installation)

1. create a personalised access token on GitLab
2. Use the created access token to git clone the repository:
   `git clone --recursive https://<user1>:<personal_access_token>@gitlab.jsc.fz-juelich.de/chew1/gsp25_particle_sfc.git`
3. change directory into the cloned repo:
   `cd gsp25_particle_sfc`
4. Make a `build` directory and change directory into it:
   `mkdir build && cd build`
5. Configure the makefiles and build:
   `cmake .. && cmake --build .`
6. The executable `p_sfc_exec` is built in `build/src` and ready to run, it is an MPI program and runs with 4 processes by default.
