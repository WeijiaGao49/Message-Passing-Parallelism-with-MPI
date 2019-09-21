/* C implementation of graphrats simulator */

#include <string.h>
#include <getopt.h>

#include "crun_seq.h"

static void full_exit(int code) {
    done(NULL);
#if MPI
    MPI_Finalize();
#endif
    exit(code);
}

static void usage(char *name) {
    char *use_string = "-g GFILE -r RFILE [-n STEPS] [-s SEED] [-q] [-i INT]";
    outmsg("Usage: %s %s\n", name, use_string);
    outmsg("   -h        Print this message\n");
    outmsg("   -g GFILE  Graph file\n");
    outmsg("   -r RFILE  Initial rat position file\n");
    outmsg("   -n STEPS  Number of simulation steps\n");
    outmsg("   -s SEED   Initial RNG seed\n");
    outmsg("   -q        Operate in quiet mode.  Do not generate simulation results\n");
    outmsg("   -i INT    Display update interval\n");
    full_exit(0);
}

int main(int argc, char *argv[]) {
    FILE *gfile = NULL;
    FILE *rfile = NULL;
    int steps = 1;
    int dinterval = 1;
    random_t global_seed = DEFAULTSEED;
    update_t update_mode = UPDATE_BATCH;
    double secs = 0.0;
    int c;
    graph_t *g = NULL;
    state_t *s = NULL;
    bool display = true;
    int process_count = 1;
    int this_zone = 0;
#if MPI
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);
    MPI_Comm_rank(MPI_COMM_WORLD, &this_zone);
#endif
    int nzone = process_count;
    bool mpi_master = this_zone == 0;
    char *optstring = "hg:r:R:n:s:i:q";
    while ((c = getopt(argc, argv, optstring)) != -1) {
        switch(c) {
        case 'h':
            if (!mpi_master) break;
            usage(argv[0]);
            break;
        case 'g':
            if (!mpi_master) break;
            gfile = fopen(optarg, "r");
            if (gfile == NULL) {
                outmsg("Couldn't open graph file %s\n", optarg);
		full_exit(1);
            }
            break;
        case 'r':
            if (!mpi_master) break;
            rfile = fopen(optarg, "r");
            if (rfile == NULL) {
                outmsg("Couldn't open rat position file %s\n", optarg);
                full_exit(1);
            }
            break;
        case 'n':
            steps = atoi(optarg);
            break;
        case 's':
            global_seed = strtoul(optarg, NULL, 0);
            break;
        case 'q':
            display = false;
            break;
        case 'i':
            dinterval = atoi(optarg);
            break;
        default:
            if (!mpi_master) break;
            outmsg("Unknown option '%c'\n", c);
            usage(argv[0]);
        }
    }

    if (mpi_master) {
      	if (gfile == NULL) {
	    outmsg("Need graph file\n");
	    usage(argv[0]);
	}
	if (rfile == NULL) {
	    outmsg("Need initial rat position file\n");
	    usage(argv[0]);
	}
	g = read_graph(gfile, nzone);
	if (g == NULL) {
	    full_exit(1);
	}
	s = read_rats(g, rfile, global_seed);
	if (s == NULL) {
	    full_exit(1);
	}
        /* Master distributes the graph to the other processors */
#if MPI
	send_graph(g);
	if (!setup_zone(g, this_zone))
	    full_exit(1);
	// TODO:
	// * Set up other zone-specific data structures
	// * Distribute copy of rats to other zones
#endif
    } else {
	/* The other nodes receive the graph from the master */
#if MPI
	g = get_graph();
	if (g == NULL) {
	    full_exit(0);
	}
	if (!setup_zone(g, this_zone))
	    full_exit(0);
	// TODO:
	// * Set up other zone-specific data structures
	// * Receive copy of rats from process 0
#endif
    }

    if (mpi_master)
	outmsg("Running with %d processes.\n", process_count);

    if (mpi_master)
	// Right now, run sequential simulator on master node
	// TODO: All processes should run simulator on their zones
        // secs = simulate(s, steps, update_mode, dinterval, display);
      secs = simulate(s, steps, update_mode, dinterval, false);
    if (mpi_master) {
	outmsg("%d steps, %d rats, %.3f seconds\n", steps, s->nrat, secs);
    }
#if MPI
    MPI_Finalize();
#endif    
    return 0;
}

