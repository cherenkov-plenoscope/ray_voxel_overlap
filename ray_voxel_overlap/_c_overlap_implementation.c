// Copyright 2017 Sebastian A. Mueller
#include <math.h>


void intersection_plane(
    double *support,
    double *direction,
    double off,
    double *intersection,
    unsigned int dim
) {
    double a = (off - support[dim])/direction[dim];
    intersection[0] = support[0] + direction[0]*a;
    intersection[1] = support[1] + direction[1]*a;
    intersection[2] = support[2] + direction[2]*a;
}

double distance(double *v, double *u) {
    double dx = v[0] - u[0];
    double dy = v[1] - u[1];
    double dz = v[2] - u[2];
    return sqrt(dx*dx + dy*dy + dz*dz);
}

double c_ray_box_overlap(
    double *support,
    double *direction,
    double xl, double xu,
    double yl, double yu,
    double zl, double zu
) {
    double hits_l[3][3];
    unsigned int nl = 0;

    double hits_u[3][3];
    unsigned int nu = 0;

    // X plane
    // -------
    if (direction[0] != 0.0) {
        double ixl[3];
        intersection_plane(support, direction, xl, ixl, 0);
        if (
            (ixl[1] >= yl && ixl[1] < yu) &&
            (ixl[2] >= zl && ixl[2] < zu)
        ) {
            nl = nl + 1;
            hits_l[nl-1][0] = ixl[0];
            hits_l[nl-1][1] = ixl[1];
            hits_l[nl-1][2] = ixl[2];
        }

        double ixu[3];
        intersection_plane(support, direction, xu, ixu, 0);
        if (
            (ixu[1] >= yl && ixu[1] <= yu) &&
            (ixu[2] >= zl && ixu[2] <= zu)
        ) {
            nu = nu + 1;
            hits_u[nu-1][0] = ixu[0];
            hits_u[nu-1][1] = ixu[1];
            hits_u[nu-1][2] = ixu[2];
        }
    }

    // Y plane
    // -------
    if (direction[1] != 0.0) {
        double iyl[3];
        intersection_plane(support, direction, yl, iyl, 1);
        if (
            (iyl[0] >= xl && iyl[0] < xu) &&
            (iyl[2] >= zl && iyl[2] < zu)
        ) {
            nl = nl + 1;
            hits_l[nl-1][0] = iyl[0];
            hits_l[nl-1][1] = iyl[1];
            hits_l[nl-1][2] = iyl[2];
        }

        double iyu[3];
        intersection_plane(support, direction, yu, iyu, 1);
        if (
            (iyu[0] >= xl && iyu[0] <= xu) &&
            (iyu[2] >= zl && iyu[2] <= zu)
        ) {
            nu = nu + 1;
            hits_u[nu-1][0] = iyu[0];
            hits_u[nu-1][1] = iyu[1];
            hits_u[nu-1][2] = iyu[2];
        }
    }

    // Z plane
    // -------
    if (direction[2] != 0.0) {
        double izl[3];
        intersection_plane(support, direction, zl, izl, 2);
        if (
            (izl[0] >= xl && izl[0] < xu) &&
            (izl[1] >= yl && izl[1] < yu)
        ) {
            nl = nl + 1;
            hits_l[nl-1][0] = izl[0];
            hits_l[nl-1][1] = izl[1];
            hits_l[nl-1][2] = izl[2];
        }

        double izu[3];
        intersection_plane(support, direction, zu, izu, 2);
        if (
            (izu[0] >= xl && izu[0] <= xu) &&
            (izu[1] >= yl && izu[1] <= yu)
        ) {
            nu = nu + 1;
            hits_u[nu-1][0] = izu[0];
            hits_u[nu-1][1] = izu[1];
            hits_u[nu-1][2] = izu[2];
        }
    }

    if       (nl == 2 && nu == 0) {
        return distance(hits_l[0], hits_l[1]);

    } else if (nl == 0 && nu == 2) {
        return distance(hits_u[0], hits_u[1]);

    } else if (nl == 1 && nu == 1) {
        return distance(hits_u[0], hits_l[0]);

    } else if (nl == 3 && nu == 3) {
        return distance(hits_u[0], hits_l[0]);

    } else if (nl == 2 && nu == 2) {
        return distance(hits_u[0], hits_l[0]);

    } else if (nl == 3 && nu == 0) {
        return sqrt((xu-xl)*(xu-xl) + (yu-yl)*(yu-yl) + (zu-zl)*(zu-zl));

    } else if (nl > 0 && nu > 0) {
        return distance(hits_u[0], hits_l[0]);

    } else if (nl == 0 && nu == 3) {
        return sqrt((xu-xl)*(xu-xl) + (yu-yl)*(yu-yl) + (zu-zl)*(zu-zl));

    } else {
        return 0.0;
    }
}

void c_next_space_partitions(
    unsigned int *dim_range,
    unsigned int dim_partitions[2][2],
    unsigned int *n
) {
    if (dim_range[1] - dim_range[0] < 1) {
        *n = *n+1;
        dim_partitions[*n-1][0] = dim_range[0];
        dim_partitions[*n-1][1] = dim_range[1];
        return;
    } else {
        unsigned int cut = (dim_range[1] - dim_range[0])/2;
        *n = *n+1;
        dim_partitions[*n-1][0] = dim_range[0];
        dim_partitions[*n-1][1] = dim_range[0] + cut;
        *n = *n+1;
        dim_partitions[*n-1][0] = dim_range[0] + cut;
        dim_partitions[*n-1][1] = dim_range[1];
        return;
    }
}

void c_overlap_of_ray_with_voxels(
    double *support,
    double *direction,
    double *x_bin_edges,
    double *y_bin_edges,
    double *z_bin_edges,
    unsigned int *x_range,
    unsigned int *y_range,
    unsigned int *z_range,

    unsigned int *number_overlaps,
    unsigned int *x_idxs,
    unsigned int *y_idxs,
    unsigned int *z_idxs,
    double *overlaps
) {
    unsigned int x_partitions[2][2];
    unsigned int nxp = 0;
    c_next_space_partitions(x_range, x_partitions, &nxp);

    unsigned int y_partitions[2][2];
    unsigned int nyp = 0;
    c_next_space_partitions(y_range, y_partitions, &nyp);

    unsigned int z_partitions[2][2];
    unsigned int nzp = 0;
    c_next_space_partitions(z_range, z_partitions, &nzp);

    for (unsigned int xp = 0; xp < nxp; xp = xp+1) {
        for (unsigned int yp = 0; yp < nxp; yp = yp+1) {
            for (unsigned int zp = 0; zp < nxp; zp = zp+1) {
                double overlap = c_ray_box_overlap(
                    support,
                    direction,
                    x_bin_edges[x_partitions[xp][0]],
                    x_bin_edges[x_partitions[xp][1]],
                    y_bin_edges[y_partitions[yp][0]],
                    y_bin_edges[y_partitions[yp][1]],
                    z_bin_edges[z_partitions[zp][0]],
                    z_bin_edges[z_partitions[zp][1]]);

                if (x_partitions[xp][1]-x_partitions[xp][0] == 1 &&
                    y_partitions[yp][1]-y_partitions[yp][0] == 1 &&
                    z_partitions[zp][1]-z_partitions[zp][0] == 1 &&
                    overlap > 0.0
                ) {
                    *number_overlaps = *number_overlaps + 1;
                    x_idxs[*number_overlaps-1] = x_partitions[xp][0];
                    y_idxs[*number_overlaps-1] = y_partitions[yp][0];
                    z_idxs[*number_overlaps-1] = z_partitions[zp][0];
                    overlaps[*number_overlaps-1] = overlap;
                } else if (overlap > 0.0) {
                    c_overlap_of_ray_with_voxels(
                        support,
                        direction,
                        x_bin_edges,
                        y_bin_edges,
                        z_bin_edges,
                        x_partitions[xp],
                        y_partitions[yp],
                        z_partitions[zp],
                        number_overlaps,
                        x_idxs,
                        y_idxs,
                        z_idxs,
                        overlaps);
                }
            }
        }
    }
    return;
}
