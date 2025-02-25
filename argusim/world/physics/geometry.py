import numpy as np
import shapely as sp
from shapely.geometry import Polygon

class Edge:
    def __init__(self, vertex1, vertex2):
        self.vertex1 = np.array(vertex1)
        self.vertex2 = np.array(vertex2)
        self.vector = self.vertex2 - self.vertex1

    def length(self):
        return np.linalg.norm(self.vector)


class Surface:
    def __init__(self, vertices):
        self.vertices = np.array(vertices)
        self.normal = self.calculate_normal()
        self.edges = self.calculate_edges(vertices)
    
    def translate(self, translation_vector):
        # Translate the surface by the given vector
        self.vertices += translation_vector
        self.edges = self.calculate_edges(self.vertices)
        
    def rotate_around_axis(self, axis_center, axis, angle):
        # Translate the surface to the origin
        translated_vertices = self.vertices - axis_center
        
        # Rotate the surface around the axis by the given angle
        rotation_matrix = self.calculate_rotation_matrix(axis, angle)
        rotated_vertices = translated_vertices @ rotation_matrix.T
        
        # Translate the surface back to the original position
        self.vertices = rotated_vertices + axis_center
        
        # Rotate the normal vector
        self.normal = self.normal @ rotation_matrix.T
        
        # Recalculate the edges
        self.edges = self.calculate_edges(self.vertices)
    
    def calculate_rotation_matrix(self, axis, angle):
        # Calculate the rotation matrix for a rotation around the axis by the given angle
        axis = np.array(axis)
        axis = axis / np.linalg.norm(axis)
        a = np.cos(angle / 2)
        b, c, d = -axis * np.sin(angle / 2)
        return np.array([
            [a ** 2 + b ** 2 - c ** 2 - d ** 2, 2 * (b * c - a * d), 2 * (b * d + a * c)],
            [2 * (b * c + a * d), a ** 2 + c ** 2 - b ** 2 - d ** 2, 2 * (c * d - a * b)],
            [2 * (b * d - a * c), 2 * (c * d + a * b), a ** 2 + d ** 2 - b ** 2 - c ** 2]
        ])

    def calculate_edges(self, vertices):
        edges = []
        num_vertices = len(vertices)
        for i in range(num_vertices):
            vertex1 = vertices[i]
            vertex2 = vertices[(i + 1) % num_vertices]
            edges.append(Edge(vertex1, vertex2))
        return edges

    def calculate_normal(self):
        # Calculate the normal vector of the surface using the first three vertices
        vector1 = self.vertices[1] - self.vertices[0]
        vector2 = self.vertices[2] - self.vertices[0]
        normal = np.cross(vector1, vector2)
        return normal / np.linalg.norm(normal)

    def is_intersecting_edge(self, point, direction, Edge):
        edge_vector = Edge.vector
        vertex1     = Edge.vertex1
        vertex2     = Edge.vertex2
        edge_normal = np.array([-edge_vector[1], edge_vector[0]])  # Perpendicular to the edge vector
        
        # Check if the point is on the same plane as the edge (always true in 2D)
        
        # Calculate the intersection point with the line of the edge
        dot_prod = np.dot(edge_normal, direction)
        
        if np.isclose(dot_prod, 0):
            # edge aligned with direction. 
            # if point is on the edge, it will be on the edge
            if np.isclose(np.dot(edge_normal, vertex1 - point), 0):
                projected_vertex1 = np.dot(vertex1 - point, direction)
                projected_vertex2 = np.dot(vertex2 - point, direction)
                # point inside edge
                if np.dot(projected_vertex1, projected_vertex2) < 0:
                    return point
                # point beyond edge
                if projected_vertex1 < 0 and projected_vertex2 < 0:
                    return False
                # vertex1 closest
                if projected_vertex1 < projected_vertex2:
                    return vertex1
                #vertex2 closest
                return vertex2
            return False
        t = np.dot(edge_normal, vertex1 - point) / dot_prod
        
        if t < 0:
            return False
        
        intersection_point = point + t * direction
        
        # Check if the intersection point is within the edge segment
        edge_length = Edge.length()
        projection_length = np.dot(intersection_point - vertex1, edge_vector) / edge_length
        
        if 0 <= projection_length <= edge_length:
            return intersection_point
        return False
        
    def is_intersecting(self, point, direction):
        # Check if a halfline starting from a point in a given direction intersects the surface
        # Calculate the vector from the point to one of the vertices of the surface
        vector_to_vertex = self.vertices[0] - point
        
        # Calculate the dot product of the direction vector and the normal vector
        dot_product = np.dot(direction, self.normal)
        
        # If the dot product is zero, the direction is parallel to the surface and does not intersect
        if np.isclose(dot_product, 0):
            return False
        # Calculate the distance along the direction vector to the intersection point
        t = np.dot(vector_to_vertex, self.normal) / dot_product
        
        # If t is negative, the intersection point is behind the starting point
        if t <= 0:
            return False
        
        # Calculate the intersection point
        intersection_point = point + t * direction
        
        # Project the intersection point and the vertices to the plane orthogonal to the normal
        # Calculate the null space of the normal vector
        null_space = np.linalg.svd(self.normal.reshape(1, -1))[2][1:].T
        
        # Project the intersection point and the vertices onto the null space
        projected_intersection = intersection_point @ null_space
        projected_vertices = self.vertices @ null_space
        projected_edges = self.calculate_edges(projected_vertices)

        # Check if the intersection point is inside the surface using the edges
        ray_trace_dir = np.array([1,0])
        intersects = []
        for edge in projected_edges:
            intersection = self.is_intersecting_edge(projected_intersection - 10.0*ray_trace_dir, ray_trace_dir, edge)
            if intersection is not False:
                intersects.append(intersection)
        # Create a vector of the x-axis of the intersects and the projected_intersection x value
        x_values = [intersect[0] for intersect in intersects] + [projected_intersection[0]]
        
        # Order the x values and store the index of the projected intersection in that ordering
        sorted_indices = np.argsort(x_values)
        
        projected_intersection_index = np.where(sorted_indices == len(intersects))[0][0]
        # Check if the intersection point is inside the surface using the vertices
        # We use the ray-casting algorithm to determine if the point is inside the polygon
        inside = False
        for i in range(len(x_values)):
            if i == projected_intersection_index and inside:
                return True
            elif i is not projected_intersection_index:
                inside = not inside
        return False
    
    def compute_unoccluded_effective_area(self, sun_direction, Body):
        # Project the vertices of the surface onto the 2D plane orthogonal to the sun direction
        sun_direction = np.array(sun_direction)
        sun_direction = sun_direction / np.linalg.norm(sun_direction)
        null_space = np.linalg.svd(sun_direction.reshape(1, -1))[2][1:].T
        
        RI2S = np.hstack((sun_direction.reshape(-1, 1), null_space))
        projected_vertices = self.vertices @ null_space
        centroid_sp = np.mean(self.vertices, 0)
        dist_from_sun = centroid_sp @ sun_direction # self.vertices @ sun_direction
        
        pol_sp = Polygon(projected_vertices) # [(0, 0), (2, 0), (2, 2), (0, 2)]
        
        # Compute the difference between the two squares
        for idx, surface in enumerate(Body.surfaces):
            Surf_vertices = surface.vertices
            projected_Surf_vertices = Surf_vertices @ null_space
            centroid_sf = np.mean(Surf_vertices, 0)
            pol_sf      = Polygon(projected_Surf_vertices)
            if centroid_sf @ sun_direction > dist_from_sun:
                for vertex in projected_Surf_vertices:
                    if np.any(np.isclose(projected_vertices, vertex, atol=1e-6)):
                        pol_sf = sp.affinity.translate(pol_sf, xoff=1e-6, yoff=1e-6)
                        break
                pol_sp2 = pol_sp.difference(pol_sf)                       
                if not pol_sp2.is_valid:
                    pol_sf = sp.affinity.translate(pol_sf, xoff=1e-6, yoff=1e-6)
                    pol_sp2 = pol_sp.difference(pol_sf)
                    if not pol_sp2.is_valid:
                        print('Invalid Polygon')
                pol_sp = pol_sp2
        
        if np.isclose(pol_sp.area, 0):
            return pol_sp.area, centroid_sp
        
        centroid = pol_sp.centroid
        # revert centroid to 3D
        centroid = np.array([centroid.x,centroid.y])
        m = (np.vstack((self.vertices[1] - self.vertices[0], self.vertices[2] - self.vertices[0])) @ RI2S).T
        b = (self.vertices[0] @ RI2S).T
        x = np.linalg.inv(m[1:]) @ (centroid - b[1:])
        centroid = RI2S @ (m @ x + b)
        
        return pol_sp.area, centroid
    

class Body:
    def __init__(self, surfaces, solar_panels=None, sensors=None):
        self.surfaces = surfaces
        self.solar_panels = solar_panels if solar_panels is not None else []
        self.sensors = sensors if sensors is not None else []
    
    def is_intersecting(self, point, direction):
        for surface in self.surfaces:
            if surface.is_intersecting(point, direction):
                return True
        return False
    
    def compute_unoccluded_effective_area(self, sun_direction, Body):
        effective_area = 0
        for solar_panel in self.solar_panels:
            effective_area += solar_panel.compute_unoccluded_effective_area(sun_direction, Body)
        return effective_area
    
    def compute_center_of_pressure(self, atm_vel_direction, Body):
        effective_areas = np.zeros(len(self.surfaces))
        centroids = np.zeros((len(self.surfaces), 3))
        for ids, surface in enumerate(self.surfaces):
            effective_areas[ids], centroids[ids] = surface.compute_unoccluded_effective_area(atm_vel_direction, Body)
        
        # compute int r x n dA = sum of the cross product of the centroid of the surface and the normal vector of the surface times effective area
        torque_factor = np.zeros(3)
        total_force_factor  = np.zeros(3)
        for ids, surface in enumerate(self.surfaces):
            # surface normal must be facing the atm_vel_direction. Then torque is in the opposite direction
            normal = surface.normal
            if np.dot(normal, atm_vel_direction) < 0:
                normal = -normal
            torque_factor += -np.cross(centroids[ids,:], normal) * effective_areas[ids]
            total_force_factor += -normal * effective_areas[ids] # needed to adjust to the CoM
        
        # T_tot = T_fac - r_com x F_fac
        # T_fac = int -r x n dA / A
        # F_fac = int -n dA / A
        total_area = np.sum(effective_areas)
        # center_of_pressure = torque_factor / total_area
        return torque_factor, total_force_factor, total_area # center_of_pressure

    def compute_sensor_visibility(self, direction):
        visibility = np.zeros(len(self.sensors), dtype=bool)
        for idx, sensor in enumerate(self.sensors):
            # Calculate the dot product with the sensor direction
            dot_product = sensor.direction @ direction

            # Calculate the angle between the direction and the sensor direction
            angle = np.degrees(np.arccos(dot_product))
            # Check visibility for each point in the view direction grid
            unoccluded  = True
            if self.is_intersecting(sensor.position, direction):
                unoccluded = False
            
            # Determine visibility based on the half angle and self occlusion
            visibility[idx] = np.logical_and(angle <= sensor.half_angle, unoccluded)
        return visibility


class Sensor():
    def __init__(self, direction, position, half_angle):
        self.position = np.array(position)
        self.direction = np.array(direction)
        self.direction = self.direction / np.linalg.norm(self.direction)
        self.position += self.direction*1e-3
        self.half_angle = half_angle


# Solar Panels need to have their facing direction defined
class SolarPanel(Surface):
    def __init__(self, vertices, facing_vector):
        super().__init__(vertices)
        self.translate(np.array(facing_vector) * 1e-3)
        self.facing_vector = np.array(facing_vector)
        self.check_facing_direction()
        self.area = self.calculate_area()

    def check_facing_direction(self):
        dot_product = np.dot(self.normal, self.facing_vector)
        if dot_product < 0:
            self.normal = -self.normal
            
    def compute_unoccluded_effective_area(self, sun_direction, Body):
        # Solar pannel effective exposed area only on one side
        dot_product = np.dot(self.normal, sun_direction)
        if dot_product < 0:
            return 0.0
        else:
            effective_area, centroid = super().compute_unoccluded_effective_area(sun_direction, Body)
            return effective_area
        
    
    def compute_effective_area(self, sun_direction):
        # Compute the effective area of the solar panel based on the angle between the sun direction and the normal
        dot_product = np.dot(self.normal, sun_direction)
        if dot_product > 0:
            return self.area * dot_product
        return 0

    def calculate_area(self):
        # Calculate the area of the solar panel
        area = 0
        num_vertices = len(self.vertices)
        for i in range(num_vertices):
            vertex1 = self.vertices[i]
            vertex2 = self.vertices[(i + 1) % num_vertices]
            area += (vertex1[0] * vertex2[1] - vertex2[0] * vertex1[1])
        return 0.5 * abs(area)