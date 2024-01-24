#include "drake/geometry/optimization/dev/fast_iris.h"
#include "drake/geometry/optimization/convex_set.h"
#include "drake/geometry/optimization/hpolyhedron.h"
#include "drake/common/parallelism.h"

using values_t = std::vector<double>;
using index_t = std::vector<uint8_t>;

index_t argsort(values_t const& values) {
    index_t index(values.size());
    std::iota(index.begin(), index.end(), 0);
    std::sort(index.begin(), index.end(), [&values](uint8_t a, uint8_t b) { return values[a] < values[b]; } );
    return index;
}


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

    const Eigen::MatrixXd ATA = options.starting_ellipse.A().T*options.starting_ellipse.A();
    const Eigen::VectorXd ellipsoid_center = options.starting_ellipse.center();

    //pre-allocate memory for the polyhedron 
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> A(
      P.A().rows() + 300, dim);

    VectorXd b(P.A().rows() + 300);

    A.topRows(P.A().rows()) = P.A();
    b.head(P.A().rows()) = P.b(); 
    std::vector<Eigen::VectorXd> particles;
    particles.reserve(options.num_particles);
    particles.emplace_back(P.UniformSample(&generator, sample));

    //populate particles by uniform sampling
    for (int i = 1; i<options.num_particles; ++i){
        particles.emplace_back(P.UniformSample(&generator, particles[i-1]));
    }
    
    int current_consecutive_failures = 0;
    int num_iterations = 0;
    
    while (current_consecutive_failures< options.num_consecutive_failures && 
           num_iterations<options.max_iterations){
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
        if(number_particles_in_collision == 0){
            ++current_consecutive_failures;
            log()->info(
            " consecutive failures: {} iteration: {} faces: {}",
            current_consecutive_failures,
            num_iterations,
            current_num_faces);
        }else{
            
            current_consecutive_failures = 0;

            //Update particles position
            std::vector<Eigen::VectorXd> particles_in_collision_updated;
            particles_in_collision_updated.reserve(number_particles_in_collision);
            
            //std::vector<Eigen::VectorXd> particles_update_distance;
            //particles_update_distance.reserve(number_particles_in_collision);

            const auto particle_update_work = [&](const int thread_num, const int64_t index) {
                const int i = static_cast<int>(index);
                auto start_point = particles_in_collision[i];
                
                Eigen::VectorXd current_point = start_point;
                
                //update particles via gradient descent and bisection
                for(int gradient_steps = 0 ; gradient_steps<options.num_gradient_steps; ++gradient_steps ){

                    // find descent direction
                    Eigen::VectorXd grad = ATA*(current_point - ellipsoid_center);
                    
                    //TODO: This might be ill conditioned, may need fixing
                    //intersection of the descent direction with the ellipsoid half axis
                    double max_distance = grad.T*ATA*(current_point- ellipsoid_center)/(grad.T*ATA*grad);

                    grad.normalize();
                    Eigen::VectorXd curr_pt_lower = current_point - max_distance*grad;
                    
                    //update current point using bisection
                    if (!checker.CheckConfigCollisionFree( curr_pt_lower, thread_num)){
                        //directly set to lowerbound
                        current_point = current_pt_lower;
                    }else{
                        // bisect to find closest point in collision
                        Eigen::VectorXd curr_pt_upper = current_point;
                        for(int i = 0; i<options.bisection_steps; ++i){
                            Eigen::MatrixXd query = 0.5*(curr_pt_upper+curr_pt_lower);
                            if (checker.CheckConfigCollisionFree( query, thread_num)){
                                // config is collision free, increase lower bound
                                curr_pt_lower = query; 
                            }else{
                                // config is in collision, decrease upper bound
                                curr_pt_upper = query; 
                            }
                            current_point = query;
                        }
                    }    
                }
                particles_in_collision_updated[i] = current_point;
            }

            //update all particles in parallel
            DynamicParallelForIndexLoop(DegreeOfParallelism(num_threads_to_use), 0,
                                        number_particles_in_collision, particle_update_work,
                                        ParallelForBackend::BEST_AVAILABLE);

            //Resample particles
            //TODO

            //Place Hyperplanes
            std::vector<double> particle_distances;
            particle_distances.reserve(number_particles_in_collision);

            for(auto particle : particles_in_collision_updated){
                particle_distances.emplace_back((particle-ellipsoid_center).T*ATA*(particle-ellipsoid_center));
            }

            //returned in ascending order
            auto indices_sorted = argsort(particle_distances);
            std::vector<bool> particle_is_redundant;
            
            for(int i = 0; i<number_particles_in_collision; ++i){
                particle_is_redundant.push_back(false);
            }

            for(auto i : indices_sorted){
                
                //add nearest face
                auto nearest_particle = particles_in_collision_updated[i];
                
                if (!particle_is_redundant[i]){
                    Eigen::VectorXd a_face = ATA*(nearest_particle - ellipsoid_center);
                    a_face.normalize();
                    double b_face = a.T*nearest_particle - options.configuration_space_margin; 
                    A.row(current_num_faces) = a_face.T;
                    b(current_num_faces) = b_face;
                    ++current_num_faces;

                    //set used particle to redundant 
                    particle_is_redundant[i] = true;
                    
                    //loop over remaining non-redundant particles and check for redundancy
                    for(int particle_index = 0; particle_index<number_particles_in_collision; ++particle_index){
                        if (!is_redundant[particle_index]){
                            if (a_face.T*particles_in_collision_updated[particle_index]-b_face >= options.configuration_space_margin + 1e-6){
                                is_redundant[particle_index] = true;
                            }
                            
                        }
                    }

                }
                
            }

            //update current polyhedron
            P = Hypolyhedron(A.topRows(current_num_faces), b.head(current_num_faces));
            
            //resample particles for next iteration
            for (int i = 1; i<options.num_particles; ++i){
                particles.emplace_back(P.UniformSample(&generator, particles[i-1]));
            }
    }
    ++num_iterations;
    }
    return P;
}


}
}
}