 #include "crun.h"

void outmsg(char *fmt, ...) {
#if MPI
    int process_id;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
    if (process_id != 0)
	fprintf(stderr, "Process %.2d|", process_id);
#endif    
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    bool got_newline = fmt[strlen(fmt)-1] == '\n';
    if (!got_newline)
	fprintf(stderr, "\n");
}

/* Allocate n int's and zero them out. 
   Maybe you could use multiple threads ... */
int *int_alloc(size_t n) {
    return (int *) calloc(n, sizeof(int));
}

/* Allocate n doubles's and zero them out.  
   Maybe you could use multiple threads ... */
double *double_alloc(size_t n) {
    return (double *) calloc(n, sizeof(double));
}

/* Allocate n random number seeds and zero them out.  */
static random_t *rt_alloc(size_t n) {
    return (random_t *) calloc(n, sizeof(random_t));
}

/* Allocate simulation state */
static state_t *new_rats(graph_t *g, int nrat, random_t global_seed) {
    int nnode = g->nnode;

    state_t *s = malloc(sizeof(state_t));
    if (s == NULL) {
	outmsg("Couldn't allocate storage for state\n");
	return NULL;
    }

    s->g = g;
    s->nrat = nrat;
    s->global_seed = global_seed;
    s->load_factor = (double) nrat / nnode;

    /* Compute batch size as max(BATCH_FRACTION * R, sqrt(R)) */
    int rpct = (int) (BATCH_FRACTION * nrat);
    int sroot = (int) sqrt(nrat);
    if (rpct > sroot)
	s->batch_size = rpct;
    else
	s->batch_size = sroot;

    // Allocate data structures
    bool ok = true;
    s->rat_position = int_alloc(nrat);
    ok = ok && s->rat_position != NULL;
    s->rat_seed = rt_alloc(nrat);
    ok = ok && s->rat_seed != NULL;
    s->rat_count = int_alloc(nnode);
    ok = ok && s->rat_count != NULL;

    s->node_weight = double_alloc(nnode);
    ok = ok && s->node_weight != NULL;
    // Allocated only when sure running in synchronous or batch mode
    s->sum_weight = NULL;
    // Only when running in synchronous or batch mode
    s->neighbor_accum_weight = NULL; 

    if (!ok) {
	outmsg("Couldn't allocate space for %d rats", nrat);
	return NULL;
    }
    return s;
}

/* Set seed values for the rats. 
   Maybe you could use multiple threads ... */
static void seed_rats(state_t *s) {
    random_t global_seed = s->global_seed;
    int nrat = s->nrat;
    int r;
    for (r = 0; r < nrat; r++) {
	random_t seeds[2];
	seeds[0] = global_seed;
	seeds[1] = r;
	reseed(&s->rat_seed[r], seeds, 2);
#if DEBUG
	if (r == TAG)
	    outmsg("Rat %d.  Setting seed to %u\n", r,
                   (unsigned) s->rat_seed[r]);
#endif
    }
}

/* See whether line of text is a comment */
static inline bool is_comment(char *s) {
    int i;
    int n = strlen(s);
    for (i = 0; i < n; i++) {
	char c = s[i];
	if (!isspace(c))
	    return c == '#';
    }
    return false;
}

/* Read in rat file */
state_t *read_rats(graph_t *g, FILE *infile, random_t global_seed) {
    char linebuf[MAXLINE];
    int r, nnode, nid, nrat;

    // Read header information
    while (fgets(linebuf, MAXLINE, infile) != NULL) {
	if (!is_comment(linebuf))
	    break;
    }
    if (sscanf(linebuf, "%d %d", &nnode, &nrat) != 2) {
	outmsg("ERROR. Malformed rat file header (line 1)\n");
	return false;
    }
    if (nnode != g->nnode) {
	outmsg("Graph contains %d nodes, but rat file has %d\n", g->nnode, nnode);
	return NULL;
    }
    
    state_t *s = new_rats(g, nrat, global_seed);


    for (r = 0; r < nrat; r++) {
	while (fgets(linebuf, MAXLINE, infile) != NULL) {
	    if (!is_comment(linebuf))
		break;
	}
	if (sscanf(linebuf, "%d", &nid) != 1) {
	    outmsg("Error in rat file.  Line %d\n", r+2);
	    return false;
	}
	if (nid < 0 || nid >= nnode) {
	    outmsg("ERROR.  Line %d.  Invalid node number %d\n", r+2, nid);
	    return false;
	}
	s->rat_position[r] = nid;
    }
    fclose(infile);

    seed_rats(s);
    outmsg("Loaded %d rats\n", nrat);
#if DEBUG
    outmsg("Load factor = %f\n", s->load_factor);
#endif
    return s;
}

/* print state of nodes */
void show(state_t *s, bool show_counts) {
    int nid;
    graph_t *g = s->g;
    printf("STEP %d %d\n", g->nnode, s->nrat);
    if (show_counts) {
            for (nid = 0; nid < g->nnode; nid++)
        	printf("%d\n", s->rat_count[nid]);
    }
    printf("END\n");
}

/* Print final output */
void done(state_t *s) {
#if MPI
    if (s == NULL || s->g->this_zone != 0)
	return;
#endif
    printf("DONE\n");
}

void init_sum_weight(state_t *s) {
    graph_t *g = s->g;
    
    if (s->sum_weight == NULL) {
	s->sum_weight = double_alloc(g->nnode);
	s->neighbor_accum_weight = double_alloc(g->nnode + g->nedge);
	if (s->sum_weight == NULL || s->neighbor_accum_weight == NULL) {
	    outmsg("Couldn't allocate space for sum_weight/neighbor_accum_weight.  Exiting");
	    exit(1);
	}

    }
}


// ------------------------------------------------------------------

void scatter_rats(              /* below are receive buffers */
                  int* my_rat_count, int** my_rats, int** my_rat_pos, random_t** my_rat_seeds,
                  int** my_rat_counts, int nNodesPerWorker,
                                /* below are send buffers */
                  int* rat_counts, int* rat_array, int* pos_array, random_t* seed_array,
                  int* rat_offsets, int* rat_count_array){

  // send first how many each proc expect to receive
  MPI_Scatter((void*) rat_counts, /* data buffer */
              1,                  /* each proc gets 1 number */
              MPI_INT,            /* data type */
              my_rat_count,       /* recv buffer */
              1,                  /* recv count */
              MPI_INT,            /* recv type */
              0,                  /* root id */
              MPI_COMM_WORLD);    /* communicator */
  
  int my_count = *my_rat_count;
  *my_rats = (int*) malloc( my_count * sizeof(int) );
  *my_rat_pos = (int*) malloc( my_count * sizeof(int) );
  *my_rat_seeds = (random_t*) malloc( my_count * sizeof(int) );
  *my_rat_counts = (int*) malloc( nNodesPerWorker * sizeof(int) );

  // send rat_array, pos_array (var length) to each worker
  MPI_Scatterv((void*) rat_array, /* data buffer */
               rat_counts,        /* size for each proc */
               rat_offsets,       /* offset for each proc */
               MPI_INT,           /* data type */
               *my_rats,          /* recv buffer */
               my_count,          /* recv count  */
               MPI_INT,           /* recv type */
               0,                 /* root id */
               MPI_COMM_WORLD);   /* communicator */
  MPI_Scatterv((void*) pos_array, /* data buffer */
               rat_counts,        /* size for each proc */
               rat_offsets,       /* offset for each proc */
               MPI_INT,           /* data type */
               *my_rat_pos,       /* recv buffer */
               my_count,          /* recv count  */
               MPI_INT,           /* recv type */
               0,                 /* root id */
               MPI_COMM_WORLD);   /* communicator */
  MPI_Scatterv((void*) seed_array, /* data buffer */
               rat_counts,         /* size for each proc */
               rat_offsets,        /* offset for each proc */
               MPI_INT,            /* data type */
               *my_rat_seeds,      /* recv buffer */
               my_count,           /* recv count  */
               MPI_INT,            /* recv type */
               0,                  /* root id */
               MPI_COMM_WORLD);    /* communicator */

  // send rat_count_array (fixed length) to each worker
  MPI_Scatter((void*) rat_count_array, /* data buffer */
              nNodesPerWorker,         /* fixed size for each proc */
              MPI_INT,                 /* data type */
              *my_rat_counts,          /* recv buffer */
              nNodesPerWorker,         /* recv count */
              MPI_INT,                 /* recv type */
              0,                       /* root id */
              MPI_COMM_WORLD);         /* communicator */

  return;
}

void prepare_rats_info(state_t* s,
                                /* below are output buffers */
                       int* rat_counts,
                       int* rat_array, int* pos_array, random_t* seed_array,
                       int* rat_offsets, int* rat_count_array){
  int i;
  int nRats = s->nrat;
  int nNodes = s->g->nnode;
  int nWorkers = s->g->nzone;
  int nNodesPerWorker = nNodes / nWorkers;
  int* rat_count = s->rat_count;
  int* rat_position = s->rat_position;
  int* zone_id = s->g->zone_id;
  random_t* rat_seed = s->rat_seed;

  // output buffers
  
  // tmp buffers
  int* rat_count_counts = (int*) calloc(nWorkers, sizeof(int));
  int* rat_count_offsets = (int*) malloc( nWorkers * sizeof(int) );

  // ~ take_census(), compute rat_count
  memset(rat_count, 0, nNodes * sizeof(int));
  for(i = 0; i < nRats; i++){
    rat_count[ rat_position[i] ]++;
  }
  
  // fill in rat_count_offsets
  for(i = 0; i < nWorkers; i++){
    rat_count_offsets[i] = i*nNodesPerWorker;
  }

  // count how many rats each worker gets
  //   and  fill in rat_count_array
  //   (rat_count_array + rat_count_offsets[i])[j] is
  //   the rat count on ith zone's jth node
  for(i = 0; i < nNodes; i++){
    int zoneId = zone_id[i];
    int curNodeCount = rat_count[i];
    rat_counts[ zoneId ] += curNodeCount;
    (rat_count_array + rat_count_offsets[ zoneId ])
      [ rat_count_counts[ zoneId ]++ ] = curNodeCount;
  }

  // fill rat_offset
  rat_offsets[0] = 0;
  for(i = 0; i < nWorkers-1; i++){
    rat_offsets[i+1] = rat_offsets[i] + rat_counts[i];
    rat_counts[i] = 0;
  }
  rat_counts[nWorkers-1] = 0;

  // fill rat_array and pos_array
  for(i = 0; i < nRats; i++){
    int nodeId = rat_position[i];
    int zoneId = zone_id[nodeId];
    (rat_array + rat_offsets[zoneId])[ rat_counts[zoneId] ] = i;
    (pos_array + rat_offsets[zoneId])[ rat_counts[zoneId] ] = nodeId;
    (seed_array + rat_offsets[zoneId])[ rat_counts[zoneId]++ ] = rat_seed[i];
  }

  // free temp storage
  free(rat_count_counts); // only master proc use this field
  free(rat_count_offsets);
  
  return;
}

void build_state(state_t* s, int my_rat_count, int my_node_count,
                 int* my_rats, int* my_rat_pos, random_t* my_rat_seeds,
                 int* my_rat_counts) {

  int i, export_total_count, import_total_count;
  int nZones = s->g->nzone;
  int localNNodes = s->g->local_node_count;
  int* rat_position = s->rat_position;
  random_t* rat_seed = s->rat_seed;
  int* rat_count = s->rat_count;
  int* my_nodes = s->g->local_node_list;
  int* export_node_count = s->g->export_node_count;
  int* import_node_count = s->g->import_node_count;
  
  // build rat mask, rat pos, rat seeds.
  char* rat_mask = (char*) calloc(s->nrat, sizeof(char));
  for(i=0; i < my_rat_count; i++){
    int curRat = my_rats[i];
    rat_mask[curRat] = 1;
    rat_position[curRat] = my_rat_pos[i];
    rat_seed[curRat] = my_rat_seeds[i];
  }
  // build rat counts (per node)
  for(i=0; i < my_node_count; i++){
    int curNode = my_nodes[i];
    rat_count[curNode] = my_rat_counts[i];
  }

  // initialize local vairiables
  s->rat_mask = rat_mask;
  s->local_nrat = my_rat_count;
  s->export_capacity = my_rat_count;
  s->import_capacity = my_rat_count;
  s->export_rat_total_count = 0;
  s->import_rat_total_count = 0;
  // allocate additional local buffers: pointer array
  s->export_rat_counts = (int*) calloc(nZones, sizeof(int));
  s->import_rat_counts = (int*) calloc(nZones, sizeof(int));
  s->export_rat_offset = (int*) malloc(nZones * sizeof(int));
  s->import_rat_offset = (int*) malloc(nZones * sizeof(int));
  // allocate additional local buffers: data array
  s->export_rat_data = (int*) malloc(my_rat_count * sizeof(int) * 3);
  /* s->export_rat_pos_data = (int*) malloc(my_rat_count * sizeof(int)); */
  /* s->export_rat_seed_data = (int*) malloc(my_rat_count * sizeof(int)); */
  s->import_rat_data = (int*) malloc(my_rat_count * sizeof(int) * 3);
  /* s->import_rat_pos_data = (int*) malloc(my_rat_count * sizeof(int)); */
  /* s->import_rat_seed_data = (int*) malloc(my_rat_count * sizeof(int)); */
  s->tmp_rat_data = (int*) malloc(my_rat_count * sizeof(int));
  // for fixed size buffers, calculate total length first
  export_total_count = 0;
  import_total_count = 0;
  for(i=0; i < nZones; i++){
    export_total_count += export_node_count[i];
    import_total_count += import_node_count[i];
  }
  
  s->send_export_count_data = (int*) malloc(export_total_count * sizeof(int));
  s->recv_export_count_data = (int*) malloc(export_total_count * sizeof(int));
  s->send_import_count_data = (int*) malloc(import_total_count * sizeof(int));
  s->recv_import_count_data = (int*) malloc(import_total_count * sizeof(int));
  s->export_weight_data = (double*) malloc(export_total_count * sizeof(double));
  s->import_weight_data = (double*) malloc(import_total_count * sizeof(double));
  // allocate request handle buffers for non-blocking sends
  s->export_count_requests = (MPI_Request*) malloc(nZones * sizeof(MPI_Request));
  s->import_count_requests = (MPI_Request*) malloc(nZones * sizeof(MPI_Request));
  s->export_weight_requests = (MPI_Request*) malloc(nZones * sizeof(MPI_Request));
  s->export_rat_requests = (MPI_Request*) malloc(nZones * sizeof(MPI_Request));
  s->import_rat_requests = (MPI_Request*) malloc(nZones * sizeof(MPI_Request));
  // allocate ``gather'' buffer
  s->local_rat_count = (int*) malloc(localNNodes * sizeof(int));
  s->gather_rat_count = (int*) malloc(localNNodes * nZones * sizeof(int));
  s->local_zone_offset = (int*) malloc(nZones * sizeof(int));

  // free received buffers
  free(my_rats);
  free(my_rat_pos);
  free(my_rat_seeds);
  free(my_rat_counts);
  return;
}

void send_rats(state_t* s){
  int nRats = s->nrat;
  int nNodes = s->g->nnode;
  int nWorkers = s->g->nzone;
  int nNodesPerWorker = nNodes / nWorkers;

  // ----prepare rat info to be send----
  int* rat_counts = (int*) calloc(nWorkers, sizeof(int));
  int* rat_offsets = (int*) malloc( nWorkers * sizeof(int) );
  int* rat_array = (int*) malloc( nRats * sizeof(int) );
  int* pos_array = (int*) malloc( nRats * sizeof(int) );
  int* rat_count_array = (int*) malloc( nNodes * sizeof(int) );
  random_t* seed_array = (random_t*) malloc( nRats * sizeof(int) );
  prepare_rats_info(s,          /* below are output buffers */
                    rat_counts, rat_array, pos_array, seed_array,
                    rat_offsets, rat_count_array);
  
  // ----broadcast global info----
  int params[2] = {s->nrat, (int) s->global_seed};
  MPI_Bcast(params, 2, MPI_INT, 0, MPI_COMM_WORLD);
  
  // ----scatter prepared info----
  // (also receive a share myself)
  int my_rat_count, *my_rats, *my_rat_pos, *my_rat_counts;
  random_t* my_rat_seeds;
  scatter_rats(                 /* below are receive params */
               &my_rat_count, &my_rats,  &my_rat_pos, &my_rat_seeds,
               &my_rat_counts, nNodesPerWorker,
                                /* below are send buffers */
               rat_counts, rat_array, pos_array, seed_array,
               rat_offsets, rat_count_array);

  // ----free local storages----
  free(rat_counts);
  free(rat_array);
  free(pos_array);
  free(seed_array);
  free(rat_offsets);
  free(rat_count_array);

  // ----build state s----
  build_state(s, my_rat_count, nNodesPerWorker,
              my_rats, my_rat_pos, (random_t*) my_rat_seeds, my_rat_counts);
  return;
}


state_t* get_rats(graph_t* g){
  // ----get global info to build s----
  int params[2];
  MPI_Bcast(params, 2, MPI_INT, 0, MPI_COMM_WORLD);
  int nrat = params[0];
  random_t global_seed = (random_t) params[1];

  // ----get scattered local rat info----
  int nNodes = g->nnode;
  int nWorkers = g->nzone;
  int nNodesPerWorker = nNodes / nWorkers;
  int my_rat_count, *my_rats, *my_rat_pos, *my_rat_counts;
  random_t* my_rat_seeds;
  scatter_rats(                 /* below are receive params */
               &my_rat_count, &my_rats,  &my_rat_pos, &my_rat_seeds,
               &my_rat_counts, nNodesPerWorker,
                                /* below are send buffers */
               NULL, NULL, NULL, NULL, NULL, NULL);

  // ----build state s----
  state_t* s = new_rats(g, nrat, global_seed);
  build_state(s, my_rat_count, nNodesPerWorker,
              my_rats, my_rat_pos, my_rat_seeds, my_rat_counts);

  return s;
}


/* Called by process 0 to collect node states from all other processes */
void gather_node_state(state_t *s) {
  int i;
  graph_t* g = s->g;
  int localNNodes = g->local_node_count;
  int nZones = g->nzone;
  int nNodes = g->nnode;
  int* send_buffer = s->local_rat_count;
  int* recv_buffer = s->gather_rat_count;
  int* rat_count = s->rat_count;
  int* zoneId = g->zone_id;
  int* local_node_list = g->local_node_list;
  for(i=0; i<localNNodes; i++){
    send_buffer[i] = rat_count[ local_node_list[i] ];
  }
  MPI_Gather((void*) send_buffer, /* buffer */
             localNNodes,         /* count */
             MPI_INT,             /* data type */
             (void*) recv_buffer, /* recv buffer */
             localNNodes,         /* recv counts */
             MPI_INT,             /* recv data type */
             0,                   /* root id */
             MPI_COMM_WORLD);     /* communicator */

  int* local_zone_offset = s->local_zone_offset;
  memset(local_zone_offset, 0, nZones*sizeof(int));
  for(i=0; i<nZones; i++){
    local_zone_offset[i] = i * localNNodes;
  }

  for(i=0; i<nNodes; i++){
    int zid = zoneId[i];
    rat_count[i] =  recv_buffer[ local_zone_offset[ zid ]++ ];
  }
  return;
}

/* Called by other processes to send their node states to process 0 */
void send_node_state(state_t *s) {
  int i;
  graph_t* g = s->g;
  int* send_buffer = s->local_rat_count;
  int localNNodes = g->local_node_count;
  int* local_node_list = g->local_node_list;
  int* rat_count = s->rat_count;
  for(i=0; i<localNNodes; i++){
    send_buffer[i] = rat_count[ local_node_list[i] ];
  }
  
  MPI_Gather((void*) send_buffer, /* buffer */
             localNNodes,         /* count */
             MPI_INT,             /* data type */
             NULL,                /* recv buffer */
             localNNodes,         /* recv counts */
             MPI_INT,             /* recv data type */
             0,                   /* root id */
             MPI_COMM_WORLD);     /* communicator */
  return;
}



/* Function suitable for sorting arrays of int's */
int comp_int(const void *ap, const void *bp) {
    int a = *(int *) ap;
    int b = *(int *) bp;
    int lt = a < b;
    int gt = a > b;
    return -lt + gt;
}

