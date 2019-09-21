#include "crun.h"


/* Compute ideal load factor (ILF) for node */
static inline double neighbor_ilf(state_t *s, int nid) {
    graph_t *g = s->g;
    int outdegree = g->neighbor_start[nid+1] - g->neighbor_start[nid] - 1;
    int *start = &g->neighbor[g->neighbor_start[nid]+1];
    int i;
    double sum = 0.0;
    for (i = 0; i < outdegree; i++) {
	int lcount = s->rat_count[nid];
	int rcount = s->rat_count[start[i]];
	double r = imbalance(lcount, rcount);
	sum += r;
    }
    double ilf = BASE_ILF + 0.5 * (sum/outdegree);
    return ilf;
}

/* Compute weight for node nid */
static inline double compute_weight(state_t *s, int nid) {
    int count = s->rat_count[nid];
    double ilf = neighbor_ilf(s, nid);
    return mweight((double) count/s->load_factor, ilf);
}


/* Recompute all node counts according to rat population */
/*
  Function only called at start of simulation, at which point
  every zone has complete rat information.  Can therefore
  have every zone update every node count.
*/
static inline void take_census(state_t *s) {
    graph_t *g = s->g;
    int nnode = g->nnode;
    int *rat_position = s->rat_position;
    int *rat_count = s->rat_count;
    int nrat = s->nrat;

    memset(rat_count, 0, nnode * sizeof(int));
    int ri;
    for (ri = 0; ri < nrat; ri++) {
	rat_count[rat_position[ri]]++;
    }
}

/* Recompute all node weights */
static inline void compute_all_weights(state_t *s) {
    int nid, i;
    graph_t *g = s->g;
    double *node_weight = s->node_weight;
    int local_nnode = g->local_node_count;
    int* local_nodes = g->local_node_list;
    // for (nid = 0; nid < g->nnode; nid++)
    for (i=0; i<local_nnode; i++){
      nid = local_nodes[i];
      node_weight[nid] = compute_weight(s, nid);
    }
}


/* In synchronous or batch mode, can precompute sums for each region */
static inline void find_all_sums(state_t *s) {
    graph_t *g = s->g;
    init_sum_weight(s);
    int nid, eid, i;
    int local_nnode = g->local_node_count;
    int* local_nodes = g->local_node_list;
    
    // for (nid = 0; nid < g->nnode; nid++) {
    for(i=0; i < local_nnode; i++){
      nid = local_nodes[i];
      double sum = 0.0;
	for (eid = g->neighbor_start[nid]; eid < g->neighbor_start[nid+1]; eid++) {
	    sum += s->node_weight[g->neighbor[eid]];
	    s->neighbor_accum_weight[eid] = sum;
	}
	s->sum_weight[nid] = sum;
    }
}

/*
  Given list of increasing numbers, and target number,
  find index of first one where target is less than list value
*/

/*
  Linear search
 */
static inline int locate_value_linear(double target, double *list, int len) {
    int i;
    for (i = 0; i < len; i++)
	if (target < list[i])
	    return i;
    /* Shouldn't get here */
    return -1;
}
/*
  Binary search down to threshold, and then linear
 */
static inline int locate_value(double target, double *list, int len) {
    int left = 0;
    int right = len-1;
    while (left < right) {
	if (right-left+1 < BINARY_THRESHOLD)
	    return left + locate_value_linear(target, list+left, right-left+1);
	int mid = left + (right-left)/2;
	if (target < list[mid])
	    right = mid;
	else
	    left = mid+1;
    }
    return right;
}


/*
  Version that can be used in synchronous or batch mode, where certain that node weights are already valid.
  And have already computed sum of weights for each node, and cumulative weight for each neighbor
  Given list of integer counts, generate real-valued weights
  and use these to flip random coin returning value between 0 and len-1
*/
static inline int fast_next_random_move(state_t *s, int r) {
    int nid = s->rat_position[r];
    graph_t *g = s->g;
    random_t *seedp = &s->rat_seed[r];
    /* Guaranteed that have computed sum of weights */
    double tsum = s->sum_weight[nid];    
    double val = next_random_float(seedp, tsum);

    int estart = g->neighbor_start[nid];
    int elen = g->neighbor_start[nid+1] - estart;
    int offset = locate_value(val, &s->neighbor_accum_weight[estart], elen);
#if DEBUG
    if (offset < 0) {
	/* Shouldn't get here */
	outmsg("Internal error.  fast_next_random_move.  Didn't find valid move.  Target = %.2f/%.2f.\n",
	       val, tsum);
	return 0;
    }
#endif
#if DEBUG
    outmsg("Computing rat %d: node %d-->%d (%.3f/%.3f)", r, nid, g->neighbor[estart+offset], val, tsum);
#endif
    return g->neighbor[estart + offset];
}

/* -------- Process single batch -------- */
static inline
void clear_neighbor_rat_count(state_t* s){
  // neighbor node == import node
  int** import_node_list = s->g->import_node_list;
  int* import_node_count = s->g->import_node_count;
  int* rat_count = s->rat_count;
  int nZones = s->g->nzone;
  int i, j;

  for(i=0; i < nZones; i++){
    int curNodeCount = import_node_count[i];
    for(j=0; j < curNodeCount; j++){
      rat_count[ import_node_list[i][j] ] = 0;
    }
  }
  return;
}

static inline
void build_and_send_int(int* send_buffer,
                        int* send_counts,
                        int** send_idx_list,
                        int* send_data,
                        MPI_Request* requests,
                        int nZones, int tag){
  int i, j;
  for(i=0; i<nZones; i++){
    int nElmToSend = send_counts[i];
    if(nElmToSend > 0){
      int* neighborNodeList = send_idx_list[i];
      // first build
      for(j=0; j<nElmToSend; j++){
        send_buffer[j] = send_data[ neighborNodeList[j] ];
      }
      // then send
      MPI_Isend((void*) send_buffer, nElmToSend, MPI_INT, i, tag,
                MPI_COMM_WORLD, requests+i);
    }// else, avoid unneccessary send/ recv
    send_buffer += nElmToSend;
  }
  return;
}

static inline
void build_and_send_double(double* send_buffer,
                           int* send_counts,
                           int** send_idx_list,
                           double* send_data,
                           MPI_Request* requests,
                           int nZones, int tag){
  int i, j;
  for(i=0; i<nZones; i++){
    int nElmToSend = send_counts[i];
    if(nElmToSend > 0){
      int* neighborNodeList = send_idx_list[i];
      // first build
      for(j=0; j<nElmToSend; j++){
        send_buffer[j] = send_data[ neighborNodeList[j] ];
      }
      // then send
      MPI_Isend((void*) send_buffer, nElmToSend, MPI_DOUBLE, i, tag,
                MPI_COMM_WORLD, requests+i);
    }// else, avoid unneccessary send/ recv
    send_buffer += nElmToSend;
  }
  return;
}



static inline
void send_escape_count(state_t* s, graph_t* g){
  // send escaped rats count:
  // -- use import_node_list (=outter nodes)
  build_and_send_int(s->send_import_count_data,
                     g->import_node_count,
                     g->import_node_list,
                     s->rat_count,
                     s->import_count_requests,
                     g->nzone, 0);
  return;
}

static inline
void send_border_count(state_t *s, graph_t* g){
  // send border rats count:
  // -- use export_node_list
  build_and_send_int(s->send_export_count_data,
                     g->export_node_count,
                     g->export_node_list,
                     s->rat_count,
                     s->export_count_requests,
                     g->nzone, 1);
  return;
}

static inline
void send_weight(state_t* s, graph_t* g){
  // send calculate weight (for next step)
  // -- use export_node_list
  build_and_send_double(s->export_weight_data,
                        g->export_node_count,
                        g->export_node_list,
                        s->node_weight,
                        s->export_weight_requests,
                        g->nzone, 3);
  return;
}


void recv_escape_count(state_t* s){
  int i, j;
  graph_t* g = s->g;
  int* rat_count = s->rat_count;
  int nZones = g->nzone;

  // receive escaped rats count
  // -- use export_node_list
  // -- add to my border rats count
  // -- also collect how many escaped rat data to recv
  
  int* recv_export_buffer = s->recv_export_count_data;
  int* export_node_count = g->export_node_count;
  int** export_node_list = g->export_node_list;
  int* import_rat_counts = s->import_rat_counts;
  int totalNeighborSum = 0;
  for(i=0; i<nZones; i++){
    int nElmRecved = export_node_count[i];
    int neighborSum = 0;
    int* borderNodeList = export_node_list[i];

    if(nElmRecved > 0){

      MPI_Recv((void*) recv_export_buffer, nElmRecved, MPI_INT, i, 0,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      
      for(j=0; j<nElmRecved; j++){
        int recvedElm = *(recv_export_buffer++);
        neighborSum += recvedElm;
        rat_count[ borderNodeList[j] ] += recvedElm;
      }
    }

    import_rat_counts[i] = neighborSum;
    totalNeighborSum += neighborSum;
  }
  s->import_rat_total_count = totalNeighborSum;

  // -- update local number of rats
  // -- check import capacity, and allocate buffers if needed.
  s->local_nrat += ( totalNeighborSum - s->export_rat_total_count );
  if(s->import_capacity < totalNeighborSum){
    free(s->import_rat_data);
    s->import_rat_data = (int*) malloc(totalNeighborSum*6*sizeof(int));
    s->import_capacity = totalNeighborSum * 2;
  }

  return;
}


void recv_border_count(state_t *s){
  int i, j;
  graph_t* g = s->g;
  int* rat_count = s->rat_count;
  int nZones = g->nzone;

  // receive border rats count
  // -- overwrites escaped rat count
  int* recv_import_buffer = s->recv_import_count_data;
  int* import_node_count = g->import_node_count;
  int** import_node_list = g->import_node_list;
  for(i=0; i<nZones; i++){
    int nElmRecved = import_node_count[i];
    if(nElmRecved > 0){
      
      MPI_Recv((void*) recv_import_buffer, nElmRecved, MPI_INT, i, 1,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      
      int* neighborNodeList = import_node_list[i];
      for(j=0; j<nElmRecved; j++){
        rat_count[ neighborNodeList[j] ] = *(recv_import_buffer++);
      }
    }
  }
  
  return;
}

void send_rat_data(state_t* s){
  int i;
  graph_t* g = s->g;
  int nZones = g->nzone;
  int* zoneId = g->zone_id;
  int nRatsToSend = s->export_rat_total_count;

  // first build rat data
  int* export_rat_offset = s->export_rat_offset;
  int* export_rat_counts = s->export_rat_counts;
  export_rat_offset[0] = 0;
  for(i=1; i<nZones; i++){
    export_rat_offset[i] = export_rat_offset[i-1] + export_rat_counts[i-1];
  }

  int* export_rats = s->tmp_rat_data;
  int* export_rat_data = s->export_rat_data;
  int* rat_position = s->rat_position;
  random_t* rat_seed = s->rat_seed;
  for(i=0; i<nRatsToSend; i++){
    int curRat = export_rats[i];
    int ratPos = rat_position[curRat];
    int ratZone = zoneId[ratPos];
    int* curDataPos = export_rat_data+( export_rat_offset[ratZone]++ )*3;
    
    curDataPos[0] = curRat;
    curDataPos[1] = ratPos;
    curDataPos[2] = (int)( rat_seed[curRat] );
  }
  
  // then send rat data
  MPI_Request* export_rat_requests = s->export_rat_requests;
  for(i=0; i<nZones; i++){
    int exportCount = export_rat_counts[i];
    
    if(exportCount > 0){
      MPI_Isend((void*) export_rat_data, /* buffer */
                exportCount*3,           /* count */
                MPI_INT,                 /* data type */
                i,                       /* dest rank (zone) */
                2,                       /* message tag */
                MPI_COMM_WORLD,          /* communicator */
                export_rat_requests+i);  /* request handle */
    }
    
    export_rat_data += exportCount*3;
  }
  return;
}


static inline
void nonblocking_recv_rat_data(state_t* s, graph_t* g){
  int i;
  int nZones = g->nzone;

  int* import_rat_counts = s->import_rat_counts;
  int* import_rat_data = s->import_rat_data;
  MPI_Request* import_rat_requests = s->import_rat_requests;
  for(i=0; i<nZones; i++){
    int importCount = import_rat_counts[i];
    if(importCount > 0){
      MPI_Irecv((void*) import_rat_data, importCount*3, MPI_INT, i, 2,
                MPI_COMM_WORLD, import_rat_requests+i);
      
    }
    import_rat_data += importCount*3;
  }

  return;
}

static inline
void update_rat_data(state_t* s, graph_t* g){
  int i, j;
  int nZones = g->nzone;
  int* import_rat_data = s->import_rat_data;
  int* import_rat_counts = s->import_rat_counts;
  char* rat_mask = s->rat_mask;
  int* rat_position = s->rat_position;
  random_t* rat_seed = s->rat_seed;
  MPI_Request* import_rat_requests = s->import_rat_requests;
  for(i=0; i<nZones; i++){
    int importCount = import_rat_counts[i];
    if(importCount > 0){

      MPI_Wait( import_rat_requests+i, MPI_STATUS_IGNORE );
      for(j=0; j<importCount; j++){
        int curRat = *(import_rat_data++);
        int ratPos = *(import_rat_data++);
        random_t ratSeed = (random_t) *(import_rat_data++);

        rat_mask[curRat] = 1;
        rat_position[curRat] = ratPos;
        rat_seed[curRat] = ratSeed;
      }
    }
  }
  return;
}

static inline
void recv_weight(state_t* s){
  int i, j;
  graph_t* g = s->g;
  int nZones = g->nzone;
  double* node_weight = s->node_weight;

  double* import_weight_data = s->import_weight_data;
  int* import_node_count = g->import_node_count;
  int** import_node_list = g->import_node_list;
  for(i=0; i<nZones; i++){
    int nElmRecved = import_node_count[i];
    if(nElmRecved > 0){
      // receive neighbor weights
      MPI_Recv((void*) import_weight_data, nElmRecved, MPI_DOUBLE, i, 3,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      // update neighbor weights
      int* neighborNodeList = import_node_list[i];
      for(j=0; j<nElmRecved; j++){
        node_weight[ neighborNodeList[j] ] = *(import_weight_data++);
      }
    }
  }
  return;
}

static inline void
wait_all_requests(state_t*s, bool wait_escape, bool wait_other){
  int i;
  int nZones = s->g->nzone;
  // wait for sending escaped rat count
  int* import_node_count = s->g->import_node_count;
  MPI_Request* import_count_requests = s->import_count_requests;
  if(wait_escape){
    for(i=0; i<nZones; i++){
      if(import_node_count[i] > 0){
        MPI_Wait(import_count_requests +i, MPI_STATUS_IGNORE);
      }
    }
  }
  // wait for sending border rat count
  // wait for sending rat data
  // wait for sending weight data
  int* export_node_count = s->g->export_node_count;
  int* export_rat_counts = s->export_rat_counts;
  MPI_Request* export_count_requests = s->export_count_requests;
  MPI_Request* export_weight_requests = s->export_weight_requests;
  MPI_Request* export_rat_requests = s->export_rat_requests;
  if(wait_other){
    for(i=0; i<nZones; i++){
      if(export_node_count[i] > 0){
        MPI_Wait(export_count_requests +i, MPI_STATUS_IGNORE);
        MPI_Wait(export_weight_requests +i, MPI_STATUS_IGNORE);
        if(export_rat_counts[i] > 0){
          MPI_Wait(export_rat_requests +i, MPI_STATUS_IGNORE);
        }
      }
    }
  }
  return;
}


static inline void do_batch(state_t *s, int batch, int bstart, int bcount) {
  int ri, escaped_nrat;
  int myZone = s->g->this_zone;
  int nZones = s->g->nzone;
  char* mask = s->rat_mask;
  int* zone_id = s->g->zone_id;
  int* escaped_rat_counts = s->export_rat_counts;
  graph_t* g = s->g;

  find_all_sums(s);

  // before simulation: 
  // -- clear rat_count of neighbor nodes
  // -- clear export_rat_counts
  // -- check export capacity (and allocate more if needed)
  clear_neighbor_rat_count(s);
  memset(escaped_rat_counts, 0, nZones * sizeof(int));
  int local_nrat = s->local_nrat;
  if(s->export_capacity < local_nrat){
    free(s->export_rat_data);
    s->export_rat_data = (int*) malloc(local_nrat * 2 * 3 * sizeof(int));
    free(s->tmp_rat_data);
    s->tmp_rat_data = (int*) malloc(local_nrat * 2 * sizeof(int));
    s->export_capacity = local_nrat * 2;
  }

  // while simulation:
  // -- update rat_count
  // -- collect a list of ``escaped'' rats  
  escaped_nrat = 0;
  int* escaped_rats = s->tmp_rat_data;
  int* rat_position = s->rat_position;
  int* rat_count = s->rat_count;
  for (ri = 0; ri < bcount; ri++) {
    int rid = ri+bstart;
    if(mask[rid]){      
      int onid = s->rat_position[rid];
      int nnid = fast_next_random_move(s, rid);
      int nzid = zone_id[nnid];
      rat_position[rid] = nnid;
      rat_count[onid] -= 1;
      rat_count[nnid] += 1;

      if( nzid !=  myZone){ // escaped!
        escaped_rats[ escaped_nrat++ ] = rid;
        escaped_rat_counts[ nzid ]++;
        mask[rid] = 0;
      }
    } // else skip (rat not in current region)
  }
  s->export_rat_total_count = escaped_nrat;

  // after simulation
  // -- first exchange escaped rat count (to get correct border count)
  // -- non-blocking send export_rat_counts and rat_data
  // -- non-blocking receive rat_data, and receive rat_count
  send_escape_count(s, g);
  recv_escape_count(s);
  send_border_count(s, g);
  send_rat_data(s);
  nonblocking_recv_rat_data(s, g);
  recv_border_count(s);

  // In the end:
  // -- update weights
  // -- exchange weight and wait for all sends
  // -- wait for receive rat_data, and update rat data
  compute_all_weights(s);
  send_weight(s, g);
  recv_weight(s);  
  wait_all_requests(s, true, true);
  update_rat_data(s, g);
}

static void batch_step(state_t *s) {
    int bstart = 0;
    int bsize = s->batch_size;
    int nrat = s->nrat;
    int bcount;
    int batch = 0;
    while (bstart < nrat) {
	bcount = nrat - bstart;
	if (bcount > bsize)
	    bcount = bsize;
	do_batch(s, batch, bstart, bcount);
	batch++;
	bstart += bcount;
    }
}


double simulate(state_t *s, int count, update_t update_mode, int dinterval, bool display) {
    int i;
    /* Compute and show initial state */
    bool show_counts = true;
    double start = currentSeconds();
    // take_census(s);

    graph_t* g = s->g;
    clear_neighbor_rat_count(s);
    send_border_count(s, g);
    send_rat_data(s);
    recv_border_count(s);
    nonblocking_recv_rat_data(s, g);
    compute_all_weights(s);
    send_weight(s, g);
    recv_weight(s);
    wait_all_requests(s, false, true);
    
    
    if (display) {
#if MPI
	if (s->g->this_zone == 0)
	    // Process 0 has a copy of the initial counts for all nodes.
	    show(s, show_counts);
        

        
#else
	show(s, show_counts);
#endif
    }
    for (i = 0; i < count; i++) {
      batch_step(s);
      if (display) {
        show_counts = (((i+1) % dinterval) == 0) || (i == count-1);
#if MPI
        if (s->g->this_zone == 0) {
          // Process 0 needs to call function show on each simulation step.
          // When show_counts is true, it will need to have
          // the counts for all other zones.
          // These must be gathered from the other processes.
          if (show_counts){
            gather_node_state(s);
          }
                
          show(s, show_counts);
        } else {
          if (show_counts){
            send_node_state(s);
          }
        }
#else
        show(s, show_counts);
#endif
      }
    }
    double delta = currentSeconds() - start;
    done(s);
    return delta;
}
