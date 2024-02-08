#include "drake/geometry/optimization/fast_iris.h"
#include <iostream>
#include <common_robotics_utilities/parallelism.hpp>

#include "drake/geometry/optimization/convex_set.h"
#include "drake/geometry/optimization/hpolyhedron.h"

using common_robotics_utilities::parallelism::DegreeOfParallelism;
using common_robotics_utilities::parallelism::DynamicParallelForIndexLoop;
using common_robotics_utilities::parallelism::ParallelForBackend;
using common_robotics_utilities::parallelism::StaticParallelForIndexLoop;
using values_t = std::vector<double>;
using index_t = std::vector<uint8_t>;

index_t argsort(values_t const& values) {
  index_t index(values.size());
  std::iota(index.begin(), index.end(), 0);
  std::sort(index.begin(), index.end(), [&values](uint8_t a, uint8_t b) {
    return values[a] < values[b];
  });
  return index;
}

namespace drake {
namespace geometry {
namespace optimization {

using math::RigidTransform;

HPolyhedron FastIris(const planning::CollisionChecker& checker,
                     const Hyperellipsoid& starting_ellipsoid,
                     const HPolyhedron& domain,
                     const FastIrisOptions& options) {

  auto start = std::chrono::high_resolution_clock::now();
  const auto parallelism = Parallelism::Max();
  const int num_threads_to_use =
      checker.SupportsParallelChecking()
          ? std::min(parallelism.num_threads(),
                     checker.num_allocated_contexts())
          : 1;
  RandomGenerator generator(options.random_seed);
  
  const Eigen::VectorXd starting_ellipsoid_center = starting_ellipsoid.center();
  
  Eigen::VectorXd current_ellipsoid_center = starting_ellipsoid.center();
  Eigen::MatrixXd current_ellipsoid_A = starting_ellipsoid.A();
  double previous_volume = 1e-12;

  const int dim = current_ellipsoid_center.size();
  int current_num_faces = domain.A().rows();
  
  DRAKE_DEMAND(domain.ambient_dimension() == dim);
  DRAKE_DEMAND(domain.IsBounded());
  DRAKE_DEMAND(domain.PointInSet(current_ellipsoid_center));

  // For debugging visualization.
  Eigen::Vector3d point_to_draw = Eigen::Vector3d::Zero();
  if (options.meshcat && dim <= 3) {
    std::string path = "seedpoint";
    options.meshcat->SetObject(path, Sphere(0.06),
                                geometry::Rgba(0.1, 1, 1, 1.0));
    point_to_draw.head(dim) = current_ellipsoid_center;
    options.meshcat->SetTransform(
        path, RigidTransform<double>(
                point_to_draw));
  }

  std::vector<Eigen::VectorXd> particles;
  particles.reserve(options.num_particles);
  //TODO: this might be overkill
  for (int i = 0; i < options.num_particles; ++i) {
    particles.emplace_back(Eigen::Vector3d::Zero());
  }
  
  int iteration = 0;
  HPolyhedron P = domain;
  HPolyhedron P_prev = domain;

  while(true){
    Eigen::MatrixXd ATA =
      current_ellipsoid_A.transpose() * current_ellipsoid_A;
    //rescaling makes max step computations more stable
    ATA = (dim/ATA.trace())*ATA;
    
    // pre-allocate memory for the polyhedron we are going to construct
    // TODO: find better soltution than hardcoding 300
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> A(
        P.A().rows() + 300, dim);
    Eigen::VectorXd b(P.A().rows() + 300);

    //initialize polytope with domain
    A.topRows(domain.A().rows()) = domain.A();
    b.head(domain.A().rows()) = domain.b();
    
    //Separating Planes Step
    int current_consecutive_failures = 0;
    int num_iterations_separating_planes = 0;
  
    while (current_consecutive_failures < options.num_consecutive_failures &&
           num_iterations_separating_planes < options.max_iterations_separating_planes) {
      
      particles.at(0) = 
        P.UniformSample(&generator, current_ellipsoid_center);
      // populate particles by uniform sampling
      for (int i = 1; i < options.num_particles; ++i) {
        particles.at(i) = P.UniformSample(&generator, particles.at(i - 1));  
      }
      
      // Find all particles in collision
      std::vector<uint8_t> particle_col_free =
          checker.CheckConfigsCollisionFree(particles, parallelism);
      std::vector<Eigen::VectorXd> particles_in_collision;
      int number_particles_in_collision = 0;
      for (size_t i = 0; i < particle_col_free.size(); ++i) {
        if (particle_col_free[i] == 0) {
          ++number_particles_in_collision;
          particles_in_collision.push_back(particles[i]);
        }
      }
  
      if (number_particles_in_collision == 0) {
        ++current_consecutive_failures;
        if (current_consecutive_failures%50 ==0 && options.verbose){
          log()->info("consecutive failures: {} iteration: {} faces: {}",
                    current_consecutive_failures, num_iterations_separating_planes,
                    current_num_faces);
        }
      } else {
        current_consecutive_failures = 0;
  
        // debugging visualization
        // if (options.meshcat && dim <= 3) {
        //   for (int pt_to_draw = 0; pt_to_draw < number_particles_in_collision;
        //        ++pt_to_draw) {
        //     std::string path = fmt::format("iteration{:02}/sepit{:02}/{:03}/initial_guess",
        //                                    iteration, num_iterations_separating_planes, pt_to_draw);
        //     options.meshcat->SetObject(path, Sphere(0.01),
        //                                geometry::Rgba(1, 0.1, 0.1, 1.0));
        //     point_to_draw.head(dim) = particles_in_collision[pt_to_draw];
        //     options.meshcat->SetTransform(
        //         path, RigidTransform<double>(point_to_draw));
        //   }
        // }
  
        // Update particles position
        std::vector<Eigen::VectorXd> particles_in_collision_updated;
        particles_in_collision_updated.reserve(number_particles_in_collision);
        for(auto p : particles_in_collision){
          particles_in_collision_updated.emplace_back(p);
        }
        // std::vector<Eigen::VectorXd> particles_update_distance;
        // particles_update_distance.reserve(number_particles_in_collision);
  
        const auto particle_update_work = [&](const int thread_num,
                                              const int64_t index) {
          const int point_idx = static_cast<int>(index);
          auto start_point = particles_in_collision[point_idx];
  
          Eigen::VectorXd current_point = start_point;
  
          // update particles via gradient descent and bisection
          // find newton descent direction
            Eigen::VectorXd grad = (current_point - current_ellipsoid_center);
            double max_distance = grad.norm();
            grad.normalize();

            Eigen::VectorXd curr_pt_lower = current_point - max_distance * grad;
            // update current point using bisection
            if (!checker.CheckConfigCollisionFree(curr_pt_lower, thread_num)) {
              // directly set to lowerbound
              current_point = curr_pt_lower;
            } else {
              // bisect to find closest point in collision
              Eigen::VectorXd curr_pt_upper = current_point;
              for (int i = 0; i < options.bisection_steps; ++i) {
                Eigen::VectorXd query = 0.5 * (curr_pt_upper + curr_pt_lower);
                if (checker.CheckConfigCollisionFree(query, thread_num)) {
                  // config is collision free, increase lower bound
                  curr_pt_lower = query;
                } else {
                  // config is in collision, decrease upper bound
                  curr_pt_upper = query;
                  current_point = query;
                }
                
              }
            }
          //}
          particles_in_collision_updated[point_idx] = current_point;
        };
        
        // update all particles in parallel
        DynamicParallelForIndexLoop(DegreeOfParallelism(num_threads_to_use), 0,
                                    number_particles_in_collision,
                                    particle_update_work,
                                    ParallelForBackend::BEST_AVAILABLE);
        // debugging visualization
        // if (options.meshcat && dim <= 3) {
           
        //   for (int pt_to_draw = 0; pt_to_draw < number_particles_in_collision;
        //        ++pt_to_draw) {
        //     std::string path = fmt::format("iteration{:02}/sepit{:02}/{:03}/updated",
        //                                    iteration, num_iterations_separating_planes, pt_to_draw);
        //     options.meshcat->SetObject(path, Sphere(0.005),
        //                                geometry::Rgba(0.5, 0.1, 0.5, 1.0));
        //     point_to_draw.head(dim) = particles_in_collision_updated[pt_to_draw];
        //     options.meshcat->SetTransform(
        //         path, RigidTransform<double>(
        //                   point_to_draw));
  
        //   //   Eigen::Matrix3Xd linepoints = Eigen::Matrix3Xd::Zero(3, 2);
        //   //   point_to_draw<<0,0,0;
        //   //   point_to_draw.head(dim) = particles_in_collision[pt_to_draw];
        //   //   linepoints.col(0) =  point_to_draw;
        //   //   point_to_draw.head(dim) = particles_in_collision_updated[pt_to_draw];
        //   //   linepoints.col(1) = point_to_draw;
            
        //   //   std::string path_line = fmt::format("iteration{:02}/{:03}/line",
        //   //                                 num_iterations_separating_planes, pt_to_draw);
        //   //   options.meshcat->SetLine(path_line, linepoints, 2.0, Rgba(0, 0, 0));
  
        //   }
        // }
        // Rresampling particles
        // TODO
  
        // Place Hyperplanes
        std::vector<double> particle_distances;
        particle_distances.reserve(number_particles_in_collision);
  
        for (auto particle : particles_in_collision_updated) {
          particle_distances.emplace_back(
              (particle - current_ellipsoid_center).transpose() * ATA *
              (particle - current_ellipsoid_center));
        }
  
        // returned in ascending order
        auto indices_sorted = argsort(particle_distances);
        std::vector<uint8_t> particle_is_redundant;
  
        for (int i = 0; i < number_particles_in_collision; ++i) {
          particle_is_redundant.push_back(0);
        }

        //add separating planes step
        int hyperplanes_added = 0;
        for (auto i : indices_sorted) {
          // add nearest face
          auto nearest_particle = particles_in_collision_updated[i];
          if (!particle_is_redundant[i]) {
            Eigen::VectorXd a_face =
                ATA * (nearest_particle - current_ellipsoid_center);
            a_face.normalize();
            double b_face = a_face.transpose() * nearest_particle -
                            options.configuration_space_margin;
            A.row(current_num_faces) = a_face.transpose();
            b(current_num_faces) = b_face;
            ++current_num_faces;
            ++hyperplanes_added;

            //resize A matrix if we need more faces
            if (A.rows()<= current_num_faces)
            {
                A.conservativeResize(A.rows() * 2, A.cols());
                b.conservativeResize(b.rows() * 2);
            }


            // debugging visualization
            if (options.meshcat && dim <= 3) {
              for (int pt_to_draw = 0; pt_to_draw < number_particles_in_collision;
                 ++pt_to_draw) {
                std::string path = fmt::format("face_pt/iteration{:02}/sepit{:02}/{:03}/pt",
                                             iteration, num_iterations_separating_planes, current_num_faces);
                options.meshcat->SetObject(path, Sphere(0.03),
                                         geometry::Rgba(1, 1, 0.1, 1.0));
                point_to_draw.head(dim) = nearest_particle;
                options.meshcat->SetTransform(
                  path, RigidTransform<double>(
                            point_to_draw));
              }
            }

            if(hyperplanes_added == options.max_separating_planes_per_iteration&&
                  options.max_separating_planes_per_iteration>0) break;
            
            if(options.verbose){
              log()->info("Face added : {} faces, iter {}",
              current_num_faces, num_iterations_separating_planes);
            }
            // set used particle to redundant
            particle_is_redundant.at(i) = true;
    
            // loop over remaining non-redundant particles and check for
            // redundancy
            #if defined(_OPENMP)
            #pragma omp parallel for num_threads(num_threads_to_use)
            #endif
              for (int particle_index = 0;
                  particle_index < number_particles_in_collision;
                  ++particle_index) {
                  if (!particle_is_redundant[particle_index]) {
                  if (a_face.transpose() *
                              particles_in_collision_updated[particle_index] -
                          b_face >=0 ) {
                      particle_is_redundant[particle_index] = 1;
                  }
                  }
              }
            }
        }
  
        // update current polyhedron
        P = HPolyhedron(A.topRows(current_num_faces), b.head(current_num_faces));
      }
      // resampling particles in current polyhedron for next iteration 
      particles[0] = P.UniformSample(&generator);
      for(int j = 1; j < options.num_particles; ++j) {
          particles[j] = P.UniformSample(&generator, particles[j-1]);
      }
      ++num_iterations_separating_planes;
    }
  
  Hyperellipsoid current_ellipsoid = P.MaximumVolumeInscribedEllipsoid();
  current_ellipsoid_A = current_ellipsoid.A();
  current_ellipsoid_center = current_ellipsoid.center();

  const double volume = current_ellipsoid.Volume();
  const double delta_volume = volume - previous_volume;
  if (delta_volume <= options.termination_threshold) {
      std::cout<<fmt::format("rel delta vol {}, thresh {}", delta_volume, options.termination_threshold)<<std::endl;
      break;
  }
  if (delta_volume / previous_volume <= options.relative_termination_threshold) {
    std::cout<<fmt::format("delta vol {}, thresh {}", delta_volume / previous_volume, options.relative_termination_threshold)<<std::endl;
      break;
  }
  ++iteration;
  if ( !(iteration < options.max_iterations)){
    std::cout<<fmt::format("iter {}, max {}", iteration, options.max_iterations)<<std::endl;
    break;
  }
  
  if (options.require_sample_point_is_contained){
    if(!(P.PointInSet(starting_ellipsoid_center))){
        std::cout<<"initial seed point not contained"<<std::endl;
        return P_prev;
    }             
  }
  previous_volume = volume;
  //reset polytope to domain, store previous iteration
  P_prev = P;
  P = domain;
  current_num_faces = P.A().rows();
  }
  auto stop = std::chrono::high_resolution_clock::now();
  if(options.verbose){
    log()->info("Fast Iris execution time : {} ms",
        std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count());
  }
  return P;
}

}  // namespace optimization
}  // namespace geometry
}  // namespace drake