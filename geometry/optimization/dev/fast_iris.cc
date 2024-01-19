#include "drake/geometry/optimization/dev/fast_iris.h"
#include "drake/geometry/optimization/convex_set.h"
#include "drake/geometry/optimization/hpolyhedron.h"

namespace drake {
namespace geometry {
namespace optimization {

HPolyhedron FastIris(
    const CollisionChecker& checker,
    const Eigen::Ref<const Eigen::VectorXd>& sample,
    const HPolyhedron& domain,
    const FastIrisOptions& options = FastIrisOptions()){

    RandomGenerator generator(options.random_seed);    
    const int dim = sample.size();
    int current_num_faces = domain.A().rows();
    HPolyhedron P = domain;

    DRAKE_DEMAND(domain.ambient_dimension() == dim);
    DRAKE_DEMAND(domain.IsBounded());

    //pre-allocate memory for the polyhedron 
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> A(
      P.A().rows() + 300, dim);

    VectorXd b(P.A().rows() + 300);

    A.topRows(P.A().rows()) = P.A();
    b.head(P.A().rows()) = P.b(); 
    Eigen::MatrixXd particles(options.num_particles, dim);

    //populate particles by uniform sampling
    particles.row(0) = P.UniformSample(&generator, sample);
    for (int i = 1; i<particles.rows(); ++i){
        particles.row(i) = P.UniformSample(&generator, particles.row(i-1));
    }

}


}
}
}