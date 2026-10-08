 !
 !===============================================================================
 ! INTERACTION SPECIFIC PARAMETERS
 !
 ! force_law                     : 3/2 = 3D-/2D-Coulomb [3] <module_interaction_specific>
 ! mac_select                    : selector for multipole acceptance criterion
 !                                 mac_select==0: Barnes-Hut [0] <module_interaction_specific>
 ! include_far_field_if_periodic : .false.==the far-field contribution to periodic boundaries is ignored
 !                                 (aka 'minimum-image-mode') [.true.] <module_interaction_specific>
 ! theta2                        : square of multipole opening angle [.36] <module_interaction_specific>
 ! eps2                          : square of short-distance cutoff parameter for Plummer potential
 !                                 (0.0 corresponds to classical Coulomb) [0.0]
 !                                 <module_interaction_specific>
 ! kelbg_invsqrttemp             : inverse square root of temperature for kelbg potential
 !                                 [0.0] <module_interaction_specific>
 !
&CALC_FORCE_COULOMB
 FORCE_LAW=3          ,
 MAC_SELECT=0          ,
 INCLUDE_FAR_FIELD_IF_PERIODIC=T,
 THETA2=  8.9999999999999997E-002,
 EPS2=  1.0000000000000000E-004,
 KELBG_INVSQRTTEMP=  0.0000000000000000     ,
 /
 !
 !===============================================================================
 ! PARAMETERS FOR LIBPEPC
 !
 ! debug_level                    : debug level for printed output [0] <module_debug>
 ! periodicity(3)                 : boolean switches to determine periodicity directions
 !                                  [.false., .false., .false.] <module_mirror_boxes>
 ! np_mult                        : a tricky parameter...
 !                                  start with -45 and decrease if crashes occur
 !                                  depends on the machine, number of particles,
 !                                  memory available, size of the tree, ...
 !                                  be careful since it increases memory
 !                                  [-45] <treevars>
 ! curve_type                     : currently has to be 1 [1] <module_libpepc_main>
 ! force_cubic_domain             : if .true. PEPC uses an overall cubic enclosure of the
 !                                  particle cloud instead of the cuboid (closer) one
 !                                  [.false.] <module_box>
 ! weighted                       : 0/1 to dis-/enable load balancing [1] <module_domains>
 ! interaction_list_length_factor : factor for increasing todo_list_length and
 !                                  defer_list_length in case of respective warning
 !                                  (e.g. for very inhomogeneous or 2D cases set to 2..8)
 !                                  [1] <treevars>
 ! mirror_box_layers              : size of near-field layer (number of shells)
 !                                  [1] <module_mirror_boxes>
 ! num_threads                    : number of threads to be used for hybrid parallelization
 !                                  (plus comm thread) [3] <treevars>
 ! idim                           : dimension of the system [3] <treevars>
 !
&LIBPEPC
 DEBUG_LEVEL=0          ,
 PERIODICITY= 3*F,
 NP_MULT= -100.000000    ,
 CURVE_TYPE=1          ,
 FORCE_CUBIC_DOMAIN=F,
 WEIGHTED=1          ,
 INTERACTION_LIST_LENGTH_FACTOR=4          ,
 MIRROR_BOX_LAYERS=1          ,
 NUM_THREADS=16         ,
 IDIM=3   ,
 /
