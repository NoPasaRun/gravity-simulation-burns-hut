# Solar System: Realistic Gravity Simulation

## Project Description

This project is a diploma thesis focused on developing a high-performance solar system simulation using modern computer graphics and physics modeling technologies. The application demonstrates realistic behavior of celestial bodies under gravitational forces, with real-time visualization.

## Key Features

- **Realistic Gravity Physics**: Implementation of the Barnes-Hut algorithm for efficient N-body problem calculation on GPU
- **High-Performance Visualization**: OpenGL with instancing for rendering thousands of objects
- **Accurate Astronomical Data**: Realistic masses, velocities, and positions of Solar System planets
- **Interactive Camera Control**: Full-featured 3D navigator with mouse and keyboard support
- **Multi-Layer Materials**: Support for day/night textures, atmosphere, and clouds for planets
- **Optional Asteroid Belt**: Generation and simulation of thousands of asteroids between Mars and Jupiter

## Technology Stack

- **Programming Language**: Python 3.12
- **Physics Simulation**: Taichi (version 1.7.4) - high-performance GPU computing library
- **Graphics Engine**: OpenGL 3.3 with GLSL shaders
- **Windowing and Input**: Pygame
- **Mathematics**: NumPy, Pyrr (matrix operations)
- **3D Models**: OBJ format with MTL material support

## Project Architecture

The project is built with a modular architecture featuring clear separation of concerns:

### Core Modules:
- **gravity.py**: Gravity engine with Barnes-Hut tree implementation on Taichi
- **render.py**: Rendering system with instancing and multi-layer material support
- **camera.py**: 3D camera control with smooth movement
- **model.py**: OBJ format 3D model loader
- **event.py**: User input handling
- **handlers.py**: Event system and commands

### Key Technical Solutions:

1. **Physics Calculation Optimization**:
   - Barnes-Hut algorithm for O(N log N) complexity instead of O(N²)
   - Fully GPU-implemented simulation on Taichi
   - Bitonic sort for efficient Morton code sorting

2. **Efficient Visualization**:
   - Instanced rendering for multiple objects
   - Shared VBO for all instance positions
   - Multi-layer shaders for different material types

3. **Scalability**:
   - Support for 100,000+ objects in simulation
   - Adaptive calculation precision based on distance

## Gravity Calculation Algorithm

The project implements the **Barnes-Hut algorithm** for efficient N-body gravitational simulation. This algorithm reduces computational complexity from O(N²) to O(N log N) by approximating distant particle groups as single points.

### Why Barnes-Hut?

Traditional N-body simulation calculates forces between every pair of particles, resulting in O(N²) complexity. For 10,000 particles, this means 100 million calculations per frame - computationally infeasible for real-time simulation. Barnes-Hut solves this by:

1. **Hierarchical Tree Structure**: Particles are organized in an octree (3D quadtree) where each node represents a region of space
2. **Center of Mass Calculation**: Each tree node stores the total mass and center of mass of all particles in its subtree
3. **Approximation Criterion**: Distant particle groups are treated as single particles if their "size" is small compared to distance

### Implementation Details

The algorithm is implemented in `gravity.py` using Taichi for GPU acceleration:

1. **Spatial Sorting with Morton Codes**:
   - Particles are assigned Morton codes based on their 3D positions
   - Morton codes interleave coordinate bits for spatial locality
   - Bitonic sort efficiently sorts particles by Morton code

2. **Tree Construction**:
   - Internal nodes represent space subdivisions
   - Leaf nodes contain individual particles
   - Parent-child relationships built using common prefix lengths of Morton codes

3. **Force Calculation with Theta Criterion**:
   - For each particle, traverse the tree recursively
   - If a node is "far enough" (node_size / distance < θ), treat as single particle
   - Otherwise, recurse into child nodes
   - Theta = 0.5 provides good balance between accuracy and performance

4. **GPU Optimization**:
   - All calculations run on GPU using Taichi kernels
   - Parallel processing of force calculations
   - Efficient memory access patterns

### Performance Benefits

- **Real-time Simulation**: Maintains 60 FPS with thousands of particles
- **Scalability**: Performance degrades gracefully with particle count
- **Accuracy**: Adaptive precision maintains realism for close interactions

## Installation and Running

### System Requirements:
- Python 3.12+
- NVIDIA GPU with Vulkan support (recommended for Taichi)
- 4GB+ RAM

### Installing Dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application:
```bash
python main.py
```

## Controls

- **Mouse**: Camera rotation
- **WASD**: Camera movement forward/backward/left/right
- **Shift/Space**: Move up/down
- **Mouse Wheel**: Zoom
- **Escape**: Exit application

## Technical Specifications

- **FPS**: 60 frames per second at 1920x1080
- **Object Count**: 9 planets + optionally up to 100,000 asteroids
- **Calculation Precision**: Double precision for positions, adaptive for forces
- **Simulation Bounds**: ±10^10 km from system center

## Future Improvements

- Adding planetary satellites (Moon, Europa, etc.)
- Implementing collisions and destructions
- Adding particle effects (meteors, comets)
- Mobile device optimization
- Web version with WebGL

## Author

Bogdan - Student, Software Developer

## License

This project is a diploma thesis and intended for educational purposes.
