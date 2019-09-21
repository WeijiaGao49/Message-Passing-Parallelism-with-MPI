#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#include "crun.h"

graph_t *new_graph(int nnode, int nedge, int nzone) {
    bool ok = true;
    graph_t *g = malloc(sizeof(graph_t));
    if (g == NULL)
	return NULL;
    g->nnode = nnode;
    g->nedge = nedge;
    g->nzone = nzone;
    g->neighbor = calloc(nnode + nedge, sizeof(int));
    ok = ok && g->neighbor != NULL;
    g->neighbor_start = calloc(nnode + 1, sizeof(int));
    ok = ok && g->neighbor_start != NULL;
#if STATIC_ILF
    g->ilf = calloc(nnode, sizeof(double));
    ok = ok && g->ilf != NULL;
#endif
    if (nzone > 0) {
	g->zone_id = calloc(nnode, sizeof(int));
	ok = ok && g->zone_id != NULL;
    }
    if (!ok) {
	outmsg("Couldn't allocate graph data structures");
	return NULL;
    }
    return g;
}

void free_graph(graph_t *g) {
    free(g->neighbor);
    free(g->neighbor_start);
    free(g);
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

/* Representation of zone */
typedef struct {
    /* Position given by upper lefthand corner */
    int x;
    int y;
    int w;
    int h;
} zone_t;

static inline int find_zone(zone_t *zones, int zone_count, int x, int y) {
    int zid;
    for (zid = 0; zid < zone_count; zid++) {
	int zxmin = zones[zid].x;
	int zymin = zones[zid].y;
	int zxmax = zxmin + zones[zid].w - 1;
	int zymax = zymin + zones[zid].h - 1;
	if (zxmin <= x && x <= zxmax && zymin <= y && y <= zymax)
	    return zid;
    }
    return -1;
}

/* Read in graph file and build graph data structure */
graph_t *read_graph(FILE *infile, int nzone) {
    char linebuf[MAXLINE];
    int nnode, nedge;
    int i, hid, tid;
    double ilf;
    int nid, eid;
    int lineno = 0;
    // How many zones does the file have?
    int fnzone = 1;

    // Read header information
    while (fgets(linebuf, MAXLINE, infile) != NULL) {
	lineno++;
	if (!is_comment(linebuf))
	    break;
    }
    if (sscanf(linebuf, "%d %d  %d", &nnode, &nedge, &fnzone) < 2) {
	outmsg("ERROR. Malformed graph file header (line 1)\n");
	return NULL;
    }

    /* See if two zone counts are compatible */
    if (fnzone % nzone != 0) {
	outmsg("ERROR.  Number of zones (%d) must be multiple of number in files (%d)",
	       nzone, fnzone);
    }
    int fzone_per_zone = fnzone / nzone;

    graph_t *g = new_graph(nnode, nedge, nzone);
    if (g == NULL)
	return g;

    nid = -1;
    // We're going to add self edges, so eid will keep track of all edges.
    eid = 0;
    for (i = 0; i < nnode; i++) {
	while (fgets(linebuf, MAXLINE, infile) != NULL) {
	    lineno++;
	    if (!is_comment(linebuf))
		break;
	}
	if (sscanf(linebuf, "n %lf", &ilf) != 1) {
	    outmsg("Line #%d of graph file malformed.  Expecting node %d\n", lineno, i+1);
	}
#if STATIC_ILF                        
	g->ilf[i] = ilf;
#endif
    }
    for (i = 0; i < nedge; i++) {
	while (fgets(linebuf, MAXLINE, infile) != NULL) {
	    lineno++;
	    if (!is_comment(linebuf))
		break;
	}
	if (sscanf(linebuf, "e %d %d", &hid, &tid) != 2) {
	    outmsg("Line #%d of graph file malformed.  Expecting edge %d\n", lineno, i+1);
	    return false;
	}
	if (hid < 0 || hid >= nnode) {
	    outmsg("Invalid head index %d on line %d\n", hid, lineno);
	    return false;
	}
	if (tid < 0 || tid >= nnode) {
	    outmsg("Invalid tail index %d on line %d\n", tid, lineno);
	    return false;
	}
	if (hid < nid) {
	    outmsg("Head index %d on line %d out of order\n", hid, lineno);
	    return false;
	    
	}
	// Starting edges for new node(s)
	while (nid < hid) {
	    nid++;
	    g->neighbor_start[nid] = eid;
	    // Self edge
	    g->neighbor[eid++] = nid;
	}
	g->neighbor[eid++] = tid;
    }
    while (nid < nnode-1) {
	// Fill out any isolated nodes
	nid++;
	g->neighbor[eid++] = nid;
    }
    g->neighbor_start[nnode] = eid;
    
    if (nzone == 0) {
	outmsg("Loaded graph with %d nodes and %d edges\n", nnode, nedge);
    } else {
	/* Space for zone information */
	zone_t zone_list[fnzone];

	for (i = 0; i < fnzone; i++) {
	    while (fgets(linebuf, MAXLINE, infile) != NULL) {
		lineno++;
		if (!is_comment(linebuf))
		    break;
	    }
	    int x, y, w, h;
	    if (sscanf(linebuf, "z %d %d %d %d", &x, &y, &w, &h) != 4) {
		outmsg("Line #%d of graph file malformed.  Expecting zone %d.\n", lineno, i+1);
	    }
	    zone_list[i].x = x; zone_list[i].y = y; zone_list[i].w = w; zone_list[i].h = h;
	}
	fclose(infile);
	/* locate nodes within zones */
	int ncol = (int) sqrt(nnode);
	for (nid = 0; nid < nnode; nid++) {
	    int x = nid % ncol;
	    int y = nid / ncol;
	    int zid = find_zone(zone_list, fnzone, x, y);
	    if (zid < 0) {
		outmsg("Error.  Could not find zone for node %d.  x = %d, y = %d", nid, x, y);
	    }
	    g->zone_id[nid] = zid / fzone_per_zone;
	    //	    outmsg("Putting node %d in graph zone %d", nid, g->zone_id[nid]);
	}
	outmsg("Loaded graph with %d nodes and %d edges (%d zones)\n", nnode, nedge, nzone);
    }


    //#if DEBUG
//    show_graph(g);
//#endif
    return g;
}

#if DEBUG
void show_graph(graph_t *g) {
    int nid, eid;
    outmsg("Graph\n");
    for (nid = 0; nid < g->nnode; nid++) {
	outmsg("%d:", nid);
	for (eid = g->neighbor_start[nid]; eid < g->neighbor_start[nid+1]; eid++) {
	    outmsg(" %d", g->neighbor[eid]);
	}
	outmsg("\n");
    }
    
}
#endif

#if MPI
/** MPI routines **/
void send_graph(graph_t *g) {
    /* Send basic graph parameters */
    int nnode = g->nnode;
    int nedge = g->nedge;
    int nzone = g->nzone;
    int params[3] = {nnode, nedge, nzone};
    MPI_Bcast(params, 3, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(g->neighbor, nedge+nnode, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(g->neighbor_start, nnode+1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(g->zone_id, nnode, MPI_INT, 0, MPI_COMM_WORLD);
}

graph_t *get_graph() {
    int params[3];
    MPI_Bcast(params, 3, MPI_INT, 0, MPI_COMM_WORLD);
    int nnode = params[0];
    int nedge = params[1];
    int nzone = params[2];
    graph_t *g = new_graph(nnode, nedge, nzone);
    if (g == NULL)
	return g;
    MPI_Bcast(g->neighbor, nedge+nnode, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(g->neighbor_start, nnode+1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(g->zone_id, nnode, MPI_INT, 0, MPI_COMM_WORLD);
    return g;
}
#endif

/*
  Given list of integers, sort and eliminate duplicates.
  Resize array and set final count
 */
static void fixup_list(int **lp, int *countp) {
    int first_count = *countp;
    int new_count = 0;
    int *list = *lp;
    qsort(list, first_count, sizeof(int), comp_int);
    int i;
    int last = -1;
    for (i = 0; i < first_count; i++) {
	int val = list[i];
	if (val != last) {
	    list[new_count++] = val;
	}
	last = val;
    }
    if (new_count == 0) {
	*lp = NULL;
	free(list);
    } else {
	*lp = realloc(list, new_count * sizeof(int));
    }
    *countp = new_count;
}

#if DEBUG
// Temporary
void format_list(int *list, int count, char *buf) {
    char *pos = buf;
    sprintf(buf, "[");
    pos += 1;
    int i;
    for (i = 0; i < count; i++) {
	if (i >= 10) {
	    sprintf(pos, " ... ]");
	    return;
	}
	if (i > 0) {
	    sprintf(pos, ", ");
	    pos += 2;
	}
	int val = list[i];
	int len = val <= 9 ? 1 : val <= 99 ? 2 : 3;
	sprintf(pos, "%d", val);
	pos += len;
    }
    sprintf(pos, "]");
}
#endif

/* Set up zone-specific data structures */
/* Return false if something goes wrong */
bool setup_zone(graph_t *g, int this_zone) {
    g->this_zone = this_zone;
    int nzone = g->nzone;
    int nnode = g->nnode;
    int maxzone = nnode/nzone;
    int maxperimeter = 4*((int) sqrt(g->nnode));
    int lcount = 0;
    int nid;
    int z;
    g->local_node_list = calloc(maxzone, sizeof(int));
    if (g->local_node_list == NULL) {
	outmsg("Couldn't allocate space for local nodes");
	return false;
    }
    g->export_node_count = calloc(nzone, sizeof(int));
    g->export_node_list = calloc(nzone, sizeof(int*));
    g->import_node_count = calloc(nzone, sizeof(int));
    g->import_node_list = calloc(nzone, sizeof(int*));
    if (g->export_node_count == NULL ||
	g->export_node_list == NULL ||
	g->import_node_count == NULL ||
	g->import_node_list == NULL) {
	outmsg("Couldn't allocate space for export/import info");
	return false;
    }
    for (z = 0; z < nzone; z++) {
	if (z != this_zone) {
	    g->export_node_list[z] = calloc(maxperimeter, sizeof(int));
	    g->import_node_list[z] = calloc(maxperimeter, sizeof(int));
	    if (g->export_node_list[z] == NULL ||
		g->import_node_list[z] == NULL) {
		outmsg("Couldn't allocate space for export/import info");
		return false;
	    }
	}
    }
    for (nid = 0; nid < nnode; nid++) {
	int zid = g->zone_id[nid];
	if (zid == this_zone) {
	    g->local_node_list[lcount++] = nid;
	    int eid;
	    for (eid = g->neighbor_start[nid]; eid < g->neighbor_start[nid+1]; eid++) {
		int nbrnid = g->neighbor[eid];
		int nbrzid = g->zone_id[nbrnid];
		if (nbrzid != this_zone) {
		    /* There's an edge from nid to nbrnid in different zone */
		    g->export_node_list[nbrzid][g->export_node_count[nbrzid]++] = nid;
		    /* and vice-versa */
		    g->import_node_list[nbrzid][g->import_node_count[nbrzid]++] = nbrnid;
		}
	    }
	}
    }
    g->local_node_count = lcount;

    for (z = 0; z < nzone; z++) {
	fixup_list(&g->export_node_list[z], &g->export_node_count[z]);
#if DEBUG
	if (g->export_node_count[z] > 0) {
	    char out_buf[100];
	    format_list(g->export_node_list[z], g->export_node_count[z], out_buf);
	    outmsg("Zone %d has %d nodes connected to zone %d: %s", this_zone, g->export_node_count[z], z, out_buf);
	}
#endif
	fixup_list(&g->import_node_list[z], &g->import_node_count[z]);
#if DEBUG
	if (g->import_node_count[z] > 0) {
	    char out_buf[100];
	    format_list(g->import_node_list[z], g->import_node_count[z], out_buf);
	    outmsg("Zone %d has %d nodes in zone %d connected to it %s", this_zone, g->import_node_count[z], z, out_buf);
	}
#endif
    }
    return true;
}
