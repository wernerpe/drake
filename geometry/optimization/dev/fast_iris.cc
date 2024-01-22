#include "drake/geometry/optimization/dev/fast_iris.h"
#include "drake/geometry/optimization/convex_set.h"
#include "drake/geometry/optimization/hpolyhedron.h"
#include "drake/common/parallelism.h"

namespace drake {
namespace geometry {
namespace optimization {

HPolyhedron FastIris(
    const CollisionChecker& checker,
    const Eigen::Ref<const Eigen::VectorXd>& sample,
    const HPolyhedron& domain,
    const FastIrisOptions& options = FastIrisOptions()){
    
    const auto parallelism = Parallelism::Max();
    RandomGenerator generator(options.random_seed);    
    const int dim = sample.size();
    int current_num_faces = domain.A().rows();
    HPolyhedron P = domain;

    DRAKE_DEMAND(domain.ambient_dimension() == dim);
    DRAKE_DEMAND(domain.IsBounded());
    DRAKE_DEMAND(domain.PointInSet(sample));

    //pre-allocate memory for the polyhedron 
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> A(
      P.A().rows() + 300, dim);

    VectorXd b(P.A().rows() + 300);

    A.topRows(P.A().rows()) = P.A();
    b.head(P.A().rows()) = P.b(); 
    std::vector<Eigen::VectorXd> particles;
    particles.push_back(P.UniformSample(&generator, sample));

    //populate particles by uniform sampling
    for (int i = 1; i<options.num_particles; ++i){
        particles.push_back(P.UniformSample(&generator, particles[i-1]));
    }
    
    //Find all particles in collision
    std::vector<uint8_t> particle_col_free = checker.CheckConfigsCollisionFree(particles, parallelism);
    std::vector<Eigen::VectorXd> particles_in_collision;
    int number_particles_in_collision = 0;
    for(size_t i = 0; i < particle_col_free.size(); ++i){
        auto element = particle_col_free[i] 
        if (element == 0){
            ++number_particles_in_collision;
            particles_in_collision.push_back(particles[i]);
        }
        
    }
    
    //Update particles position

    //Resample particles

    //Place Hyperplanes

    return P;
}


}
}
}