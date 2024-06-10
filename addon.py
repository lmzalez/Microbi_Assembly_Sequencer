import bpy
import bmesh
import heapq
import string
import random
import math
import mathutils
from mathutils import Vector

print("addon.py is being imported")

# Update the size of the text displayed on faces in the 3D view
def update_face_text_size(self, context):
    print(f"Updating face text size to {context.scene.microbi_face_text_size}")
    for obj in context.scene.objects:
        if obj.type == 'FONT' and obj.name.startswith("Face_"):
            print(f"Updating {obj.name} to size {context.scene.microbi_face_text_size}")
            obj.data.size = context.scene.microbi_face_text_size

# Update the size of the text displayed on edges in the 3D view
def update_edge_text_size(self, context):
    print(f"Updating edge text size to {context.scene.microbi_edge_text_size}")
    for obj in context.scene.objects:
        if obj.type == 'FONT' and obj.name.startswith("Edge_"):
            print(f"Updating {obj.name} to size {context.scene.microbi_edge_text_size}")
            obj.data.size = context.scene.microbi_edge_text_size

# Property group to store face data, including index, centroid, and transformation matrix
class FaceDataPropertyGroup(bpy.types.PropertyGroup):
    face_index: bpy.props.IntProperty()
    centroid: bpy.props.FloatVectorProperty()
    matrix: bpy.props.FloatVectorProperty(size=16)

# Property group to store sorted face indices and their new order
class SortedFaceIndexPropertyGroup(bpy.types.PropertyGroup):
    face_index: bpy.props.IntProperty()
    new_index: bpy.props.IntProperty()

# Add empty objects to the faces of a mesh to mark their centroids
def add_face_empties(obj, empty_display_size):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)

    face_data = {}
    for face in bm.faces:
        normal = face.normal.normalized()
        if len(face.verts) >= 2:
            tangent = (face.verts[1].co - face.verts[0].co).normalized()
        else:
            tangent = Vector((1, 0, 0))
        bitangent = normal.cross(tangent).normalized()
        transform_matrix = mathutils.Matrix((tangent, bitangent, normal)).transposed()
        transform_matrix.resize_4x4()
        transform_matrix.translation = obj.matrix_world @ face.calc_center_median()
        empty = bpy.data.objects.new("Empty", None)
        empty.empty_display_size = empty_display_size
        empty.matrix_world = transform_matrix
        bpy.context.collection.objects.link(empty)
        face_data[face.index] = (empty, face.calc_center_median())

    bpy.ops.object.mode_set(mode='OBJECT')
    return face_data

# Update the face data properties with new centroids and transformation matrices
def update_face_data(context, obj, face_data_prop):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)

    bm.faces.ensure_lookup_table()

    for item in face_data_prop:
        face = bm.faces[item.face_index]
        item.centroid = (obj.matrix_world @ face.calc_center_median())
        normal = face.normal.normalized()
        tangent = (face.verts[1].co - face.verts[0].co).normalized() if len(face.verts) >= 2 else Vector((1, 0, 0))
        bitangent = normal.cross(tangent).normalized()
        transform_matrix = mathutils.Matrix((tangent, bitangent, normal)).transposed()
        transform_matrix.resize_4x4()
        transform_matrix.translation = obj.matrix_world @ face.calc_center_median()
        item.matrix = [val for row in transform_matrix for val in row]

    bpy.ops.object.mode_set(mode='OBJECT')

# Label the edges of a mesh based on the sorted face indices
def edge_labeling(bm, sorted_face_indices):
    edge_labels = {}
    edge_faces = {}
    alphabet = string.ascii_uppercase

    for edge in bm.edges:
        linked_faces = edge.link_faces
        if len(linked_faces) == 2:
            face1, face2 = linked_faces
            face1_index = sorted_face_indices[face1.index]
            face2_index = sorted_face_indices[face2.index]
            label = f"{face1_index}_{alphabet[edge.index % 26]}_{face2_index}"
            edge_labels[edge.index] = label
            edge_faces[edge.index] = [face1.index, face2.index]
        elif len(linked_faces) == 1:
            face = linked_faces[0]
            face_index = sorted_face_indices[face.index]
            label = f"{face_index}_{alphabet[edge.index % 26]}"
            edge_labels[edge.index] = label
            edge_faces[edge.index] = [face.index]

    return edge_labels, edge_faces

# Operator to set a seam for UV unwrapping in edit mode
class MICROBI_OT_set_seam(bpy.types.Operator):
    bl_idname = "object.microbi_set_seam"
    bl_label = "Set Seam (Edit Mode)"
    bl_description = "Set a seam for the UV unwrap"

    @classmethod
    def poll(cls, context):
        return (context.active_object is not None and
                context.active_object.type == 'MESH' and
                context.active_object.mode == 'EDIT')

    def execute(self, context):
        bpy.ops.object.mode_set(mode='OBJECT')

        mesh = bpy.context.active_object.data
        for edge in mesh.edges:
            if edge.select:
                edge.use_seam = True

        bpy.ops.object.mode_set(mode='EDIT')
        return {'FINISHED'}

# Operator to clear selected seams on the active mesh in edit mode
class MICROBI_OT_clear_seam(bpy.types.Operator):
    bl_idname = "object.microbi_clear_seam"
    bl_label = "Clear Seam (Edit Mode)"
    bl_description = "Clear selected seams on the active mesh"

    @classmethod
    def poll(cls, context):
        return (context.active_object is not None and
                context.active_object.type == 'MESH' and
                context.active_object.mode == 'EDIT')

    def execute(self, context):
        bpy.ops.object.mode_set(mode='OBJECT')

        mesh = bpy.context.active_object.data
        for edge in mesh.edges:
            if edge.select:
                edge.use_seam = False

        bpy.ops.object.mode_set(mode='EDIT')
        return {'FINISHED'}

# Operator to UV unwrap the mesh and create a 2D mesh layout
class MICROBI_OT_uv_unwrap(bpy.types.Operator):
    bl_idname = "object.microbi_uv_unwrap"
    bl_label = "UV Unwrap"
    bl_description = "UV unwrap and visualize 2D mesh while keeping original in place"

    def execute(self, context):
        def create_uv_mesh_representation_without_empties(orig_obj, sample_size=3, additional_scale=1.3):
            context = bpy.context
            context.view_layer.objects.active = orig_obj
            orig_obj.select_set(True)

            bpy.ops.object.duplicate()
            uv_obj = context.object
            uv_obj.name = orig_obj.name + "_UV_Layout"

            context.view_layer.objects.active = uv_obj
            bpy.ops.object.mode_set(mode='EDIT')
            bm_uv = bmesh.from_edit_mesh(uv_obj.data)
            uv_layer_uv = bm_uv.loops.layers.uv.verify()
            bpy.ops.mesh.select_all(action='DESELECT')
            for e in bm_uv.edges:
                if e.seam:
                    e.select = True
            bpy.ops.mesh.edge_split()

            edges = [e for e in bm_uv.edges if not e.seam]
            selected_edges = random.sample(edges, min(sample_size, len(edges)))
            total_ratio = 0
            for edge in selected_edges:
                loop = next(l for l in edge.link_loops)
                uv1 = loop[uv_layer_uv].uv
                uv2 = loop.link_loop_next[uv_layer_uv].uv
                uv_dist = (uv1 - uv2).length
                real_dist = edge.calc_length()
                total_ratio += real_dist / uv_dist if uv_dist != 0 else 0

            scale_factor = (total_ratio / len(selected_edges)) * additional_scale

            uv_centroid = Vector((0, 0, 0))
            for vert in bm_uv.verts:
                uv_centroid += vert.co
            uv_centroid /= len(bm_uv.verts)

            for vert in bm_uv.verts:
                vert.co -= uv_centroid

            for face in bm_uv.faces:
                for loop in face.loops:
                    uv = loop[uv_layer_uv].uv
                    loop.vert.co = Vector((uv.x, uv.y, 0)) * scale_factor

            new_uv_centroid = Vector((0, 0, 0))
            for vert in bm_uv.verts:
                new_uv_centroid += vert.co
            new_uv_centroid /= len(bm_uv.verts)

            for vert in bm_uv.verts:
                vert.co -= new_uv_centroid

            bmesh.update_edit_mesh(uv_obj.data)
            bpy.ops.object.mode_set(mode='OBJECT')
            uv_obj.display_type = 'WIRE'

            face_data_uv = collect_face_data(uv_obj)
            face_data_orig = collect_face_data(orig_obj)

            for empty, _ in face_data_uv.values():
                if isinstance(empty, bpy.types.Object):
                    bpy.data.objects.remove(empty, do_unlink=True)
            for empty, _ in face_data_orig.values():
                if isinstance(empty, bpy.types.Object):
                    bpy.data.objects.remove(empty, do_unlink=True)

            context.scene.face_data_3d.clear()
            context.scene.face_data_2d.clear()
            for index, (matrix, centroid) in face_data_orig.items():
                item = context.scene.face_data_3d.add()
                item.face_index = index
                item.matrix = [element for row in matrix for element in row]
                item.centroid = centroid
            for index, (matrix, centroid) in face_data_uv.items():
                item = context.scene.face_data_2d.add()
                item.face_index = index
                item.matrix = [element for row in matrix for element in row]
                item.centroid = centroid

            return face_data_orig, face_data_uv, uv_obj

        def collect_face_data(obj):
            context.view_layer.objects.active = obj
            bpy.ops.object.mode_set(mode='EDIT')
            bm = bmesh.from_edit_mesh(obj.data)

            face_data = {}
            for face in bm.faces:
                normal = face.normal.normalized()
                if len(face.verts) >= 2:
                    tangent = (face.verts[1].co - face.verts[0].co).normalized()
                else:
                    tangent = Vector((1, 0, 0))
                bitangent = normal.cross(tangent).normalized()
                transform_matrix = mathutils.Matrix((tangent, bitangent, normal)).transposed()
                transform_matrix.resize_4x4()
                transform_matrix.translation = obj.matrix_world @ face.calc_center_median()
                face_data[face.index] = (transform_matrix, face.calc_center_median())

            bpy.ops.object.mode_set(mode='OBJECT')
            return face_data

        obj = context.active_object

        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

        face_data_orig, face_data_uv, uv_obj = create_uv_mesh_representation_without_empties(obj)

        context.view_layer.objects.active = uv_obj
        for obj in context.selected_objects:
            obj.select_set(False)
        uv_obj.select_set(True)

        return {'FINISHED'}

# Operator to rotate the mesh 90 degress around the z-axis
class MICROBI_OT_rotate_90(bpy.types.Operator):
    bl_idname = "object.microbi_rotate_90"
    bl_label = "Rotate 90 Degrees"
    bl_description = "Rotate the mesh 90 degrees around the object's origin"

    def execute(self, context):
        obj = context.active_object

        bpy.ops.object.mode_set(mode='OBJECT')

        bpy.ops.ed.undo_push(message="Pre-Rotation")

        obj.rotation_euler[2] += math.radians(90)

        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

        update_face_data(context, obj, context.scene.face_data_2d)

        bpy.ops.ed.undo_push(message="Post-Rotation")

        return {'FINISHED'}

# Operator to create a minimum spanning tree (MST) based on the faces of the mesh
class MICROBI_OT_create_mst(bpy.types.Operator):
    bl_idname = "object.microbi_create_mst"
    bl_label = "Create MST"
    bl_description = "Sort faces based on a minimum spanning tree and create a visual representation"

    def execute(self, context):
        import bmesh
        from mathutils import Vector
        import heapq
        import string

        def setup_grease_pencil(collection):
            if "GPencil" not in bpy.data.objects:
                gp_data = bpy.data.grease_pencils.new("GPencil")
                gpencil = bpy.data.objects.new(gp_data.name, gp_data)
                collection.objects.link(gpencil)
                layer = gp_data.layers.new(name="MST_Layer", set_active=True)
            else:
                gpencil = bpy.data.objects["GPencil"]
                layer = gpencil.data.layers.get("MST_Layer") or gpencil.data.layers.new(name="MST_Layer")
            return gpencil, layer

        def add_gp_material(gpencil):
            material_name = "GP_Material"
            if material_name not in bpy.data.materials:
                gp_material = bpy.data.materials.new(name=material_name)
                bpy.data.materials.create_gpencil_data(gp_material)
                gp_material.grease_pencil.color = (0.94, 0.99, 0.91, 1)
                gp_material.grease_pencil.show_stroke = True
            else:
                gp_material = bpy.data.materials[material_name]

            if gp_material.name not in [mat.name for mat in gpencil.data.materials]:
                gpencil.data.materials.append(gp_material)
            gpencil.active_material = gp_material

            return gpencil.data.materials.find(gp_material.name)

        def cool_colormap(num_colors):
            colors = []
            for i in range(num_colors):
                r = i / (num_colors - 1)
                g = 1.0 - r
                b = 1.0
                colors.append((r, g, b, 1.0))
            return colors

        def draw_grease_pencil_line(gp_layer, p1, p2, material_index, color):
            if gp_layer.frames:
                frame = gp_layer.frames[0]
            else:
                frame = gp_layer.frames.new(1)

            stroke = frame.strokes.new()
            stroke.display_mode = '3DSPACE'
            stroke.points.add(count=2)
            stroke.points[0].co = p1
            stroke.points[1].co = p2
            stroke.material_index = material_index

            for point in stroke.points:
                point.vertex_color = color

        def create_text_at(location, text, collection, context, is_face_text=True):
            text_size = context.scene.microbi_face_text_size if is_face_text else context.scene.microbi_edge_text_size
            bpy.ops.object.text_add(location=location)
            text_obj = bpy.context.object
            text_obj.data.body = str(text)
            text_obj.data.align_x = 'CENTER'
            text_obj.data.align_y = 'CENTER'
            text_obj.data.size = text_size
            text_obj.name = f"Face_{text}" if is_face_text else f"Edge_{text}_Label"
            collection.objects.link(text_obj)
            bpy.context.collection.objects.unlink(text_obj)
            return text_obj

        def prim(start, adjacency_list, gp_layer):
            mst, visited, edge_map, dfs_order = [], set(), {}, []
            min_heap = [(0, start, -1)]
            while min_heap:
                cost, node, from_node = heapq.heappop(min_heap)
                if node not in visited:
                    visited.add(node)
                    if from_node != -1:
                        mst.append((from_node, node))
                        if from_node not in edge_map:
                            edge_map[from_node] = []
                        if node not in edge_map:
                            edge_map[node] = []
                        edge_map[from_node].append(node)
                        edge_map[node].append(from_node)
                        dfs_order.append((from_node, node))
                    for next_cost, neighbor in adjacency_list[node]:
                        if neighbor not in visited:
                            heapq.heappush(min_heap, (next_cost, neighbor, node))
            return mst, edge_map, dfs_order

        def weighted_dfs(node, edge_map, centroids, collection, context, visited=None, index=1, index_map=None, dfs_order=None):
            if visited is None:
                visited = set()
            if index_map is None:
                index_map = {}
            if dfs_order is None:
                dfs_order = []

            visited.add(node)
            index_map[node] = index
            dfs_order.append(node)
            create_text_at(centroids[node], index, collection, context, is_face_text=True)

            item = context.scene.sorted_face_indices.add()
            item.face_index = node
            item.new_index = index

            if node in edge_map:
                children = sorted(edge_map[node], key=lambda x: centroids[x].x, reverse=True)
                for child in children:
                    if child not in visited:
                        index = weighted_dfs(child, edge_map, centroids, collection, context, visited, index + 1, index_map, dfs_order)
            return index

        def process_mesh(obj, colormap_func, total_faces, context, start_index=1):
            if obj is None or obj.type != 'MESH':
                raise ValueError("Selected object is not a mesh or no object is selected")

            bpy.ops.object.mode_set(mode='OBJECT')
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

            bm = bmesh.new()
            bm.from_mesh(obj.data)

            if "MST_Visualization" not in bpy.data.collections:
                mst_collection = bpy.data.collections.new("MST_Visualization")
                bpy.context.scene.collection.children.link(mst_collection)
            else:
                mst_collection = bpy.data.collections["MST_Visualization"]

            gpencil, gp_layer = setup_grease_pencil(mst_collection)
            material_index = add_gp_material(gpencil)

            centroids = {f.index: f.calc_center_median() for f in bm.faces}
            adjacency_list = {f.index: [] for f in bm.faces}
            for edge in bm.edges:
                linked_faces = list(edge.link_faces)
                if len(linked_faces) == 2:
                    f1, f2 = linked_faces
                    weight = (centroids[f1.index] - centroids[f2.index]).length
                    adjacency_list[f1.index].append((weight, f2.index))
                    adjacency_list[f2.index].append((weight, f1.index))

            start_face = min(centroids.keys(), key=lambda i: (centroids[i].x, centroids[i].y))

            print("Running Prim's Algorithm")
            mst, edge_map, dfs_order = prim(start_face, adjacency_list, gp_layer)
            print(f"mst: {mst}, edge_map: {edge_map}, dfs_order: {dfs_order}")

            new_indices = {}
            print("Running Weighted DFS")
            final_index = weighted_dfs(start_face, edge_map, centroids, mst_collection, context, index=start_index, index_map=new_indices, dfs_order=dfs_order)
            print(f"new_indices: {new_indices}, dfs_order: {dfs_order}")

            filtered_dfs_order = [pair for pair in dfs_order if isinstance(pair, tuple)]
            print(f"Filtered dfs_order: {filtered_dfs_order}")

            num_faces = total_faces
            colors = colormap_func(num_faces)
            for from_node, to_node in filtered_dfs_order:
                factor = (new_indices[to_node] - 1) / (num_faces - 1) if num_faces > 1 else 0
                color = colors[int(factor * (num_faces - 1))]
                draw_grease_pencil_line(gp_layer, centroids[from_node], centroids[to_node], material_index, color)

            bm.to_mesh(obj.data)
            bm.free()
            obj.data.update()

            return final_index

        selected_objects = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
        if len(selected_objects) == 0:
            raise ValueError("No mesh objects selected")
        elif len(selected_objects) == 1:
            process_mesh(selected_objects[0], cool_colormap, total_faces=len(selected_objects[0].data.polygons), context=context)
        else:
            active_object = bpy.context.view_layer.objects.active
            if active_object not in selected_objects:
                raise ValueError("Active object is not in the selection")

            total_faces = sum(len(obj.data.polygons) for obj in selected_objects)
            next_start_index = process_mesh(active_object, cool_colormap, total_faces=total_faces, context=context)
            for obj in selected_objects:
                if obj != active_object:
                    next_start_index = process_mesh(obj, cool_colormap, total_faces=total_faces, start_index=next_start_index + 1, context=context)

        return {'FINISHED'}

# Operator to toggle the visibility of the Grease Pencil object for the MST
class MICROBI_OT_toggle_gpencil_visibility(bpy.types.Operator):
    bl_idname = "object.microbi_toggle_gpencil_visibility"
    bl_label = "Show/Hide"
    bl_description = "Toggle the visibility of the Grease Pencil object"

    def execute(self, context):
        gpencil_objects = [obj for obj in bpy.data.objects if obj.type == 'GPENCIL']
        if gpencil_objects:
            gpencil = gpencil_objects[0]
            gpencil.hide_viewport = not gpencil.hide_viewport
            gpencil.hide_render = gpencil.hide_viewport
            self.report({'INFO'}, f"GPencil visibility set to: {not gpencil.hide_viewport}")
        else:
            self.report({'WARNING'}, "No GPencil object found")
        return {'FINISHED'}

# Operator to separate loose components of the mesh into individual parts and reset their origins
class MICROBI_OT_create_loose_parts(bpy.types.Operator):
    bl_idname = "object.microbi_create_loose_parts"
    bl_label = "Create Loose Parts"
    bl_description = "Separates loose components into parts and resets their origin"

    @classmethod
    def poll(cls, context):
        return context.active_object is not None and context.active_object.type == 'MESH'

    def execute(self, context):
        obj = context.active_object
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')

        bpy.ops.mesh.separate(type='LOOSE')

        bpy.ops.object.mode_set(mode='OBJECT')

        for o in context.selected_objects:
            if o != obj:
                bpy.context.view_layer.objects.active = o
                bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')

        return {'FINISHED'}

# Operator to reorient components from the 3D mesh to the 2D mesh layout
class MICROBI_OT_sort_components(bpy.types.Operator):
    bl_idname = "object.microbi_sort_components"
    bl_label = "Sort Components"
    bl_description = "Reorients components from the 3d mesh to the 2d mesh layout"

    def execute(self, context):
        def adjust_parent_to_2d_mesh(component, face_data_3d, face_data_2d):
            component_centroid = component.location
            closest_data_3d = min(face_data_3d.values(), key=lambda x: (Vector(x.centroid) - component_centroid).length)
            closest_matrix_3d = mathutils.Matrix([closest_data_3d.matrix[i:i + 4] for i in range(0, 16, 4)])
            face_index_3d = [index for index, data in face_data_3d.items() if Vector(data.centroid) == Vector(closest_data_3d.centroid)][0]

            if face_index_3d in face_data_2d:
                closest_data_2d = face_data_2d[face_index_3d]
                closest_matrix_2d = mathutils.Matrix([closest_data_2d.matrix[i:i + 4] for i in range(0, 16, 4)])

                try:
                    inverted_matrix = closest_matrix_3d.inverted()
                    transform_matrix = closest_matrix_2d @ inverted_matrix
                    component.matrix_world = transform_matrix @ component.matrix_world
                except ValueError:
                    print(f"Error processing component '{component.name}': Non-invertible matrix")

        def rename_components(context, components):
            sorted_face_indices = {item.face_index: item.new_index for item in context.scene.sorted_face_indices}
            for component in components:
                component_centroid = component.location
                closest_face_index = min(sorted_face_indices.keys(), key=lambda x: (Vector(face_data_2d[x].centroid) - component_centroid).length)
                component.name = str(sorted_face_indices[closest_face_index])

        components = [obj for obj in context.selected_objects if obj.type == 'MESH']

        face_data_3d = {item.face_index: item for item in context.scene.face_data_3d}
        face_data_2d = {item.face_index: item for item in context.scene.face_data_2d}

        if not face_data_3d or not face_data_2d:
            self.report({'ERROR'}, "Ensure you have created the UV Layout using UV Unwrap")
            return {'CANCELLED'}

        for component in components:
            adjust_parent_to_2d_mesh(component, face_data_3d, face_data_2d)

        rename_components(context, components)

        bpy.ops.ed.undo_push(message="Sort Components")

        bpy.ops.object.select_all(action='DESELECT')
        for component in components:
            component.select_set(True)
        context.view_layer.objects.active = components[0] if components else None
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

        return {'FINISHED'}

# Operator to label the edges of the selected mesh
class MICROBI_OT_edge_naming(bpy.types.Operator):
    bl_idname = "object.microbi_edge_naming"
    bl_label = "Edge Naming"
    bl_description = "Labels the edges of the selected mesh based on the operator logic"

    def execute(self, context):
        obj = context.active_object

        if obj is None or obj.type != 'MESH':
            self.report({'ERROR'}, "No mesh object selected")
            return {'CANCELLED'}

        bpy.ops.object.mode_set(mode='EDIT')
        bm = bmesh.from_edit_mesh(obj.data)
        bm.faces.ensure_lookup_table()
        bm.edges.ensure_lookup_table()

        sorted_face_indices = {item.face_index: item.new_index for item in context.scene.sorted_face_indices}
        edge_labels, edge_faces = edge_labeling(bm, sorted_face_indices)

        bpy.ops.object.mode_set(mode='OBJECT')

        print(f"Edge labels: {edge_labels}")

        bpy.context.view_layer.objects.active = None
        bpy.ops.object.select_all(action='DESELECT')
        bpy.context.view_layer.objects.active = obj

        for edge in obj.data.edges:
            edge_vector = (obj.data.vertices[edge.vertices[1]].co - obj.data.vertices[edge.vertices[0]].co)
            edge_mid_point = (obj.data.vertices[edge.vertices[0]].co + obj.data.vertices[edge.vertices[1]].co) / 2

            for face_index in edge_faces[edge.index]:
                sorted_face_index = sorted_face_indices.get(face_index, face_index)
                face_center = obj.data.polygons[face_index].center
                offset_distance = (face_center - edge_mid_point).length * 0.1
                offset_direction = (face_center - edge_mid_point).normalized() * offset_distance
                offset_mid_point = edge_mid_point + offset_direction

                bpy.ops.object.text_add(location=obj.matrix_world @ offset_mid_point)
                edge_text_obj = context.view_layer.objects.active
                label = edge_labels[edge.index]

                label_parts = label.split('_')
                if int(label_parts[0]) == sorted_face_index:
                    edge_text_obj.data.body = label
                else:
                    edge_text_obj.data.body = f"{label_parts[2]}_{label_parts[1]}_{label_parts[0]}"

                edge_text_obj.data.size = context.scene.microbi_edge_text_size
                edge_text_obj.data.align_x = 'CENTER'
                edge_text_obj.data.align_y = 'CENTER'
                edge_text_obj.rotation_mode = 'XYZ'
                global_x = Vector((1, 0, 0))
                edge_angle = global_x.angle(edge_vector)
                edge_text_obj.rotation_euler = (0, 0, edge_angle if edge_vector.y >= 0 else -edge_angle)
                edge_text_obj.name = f"Edge_{edge.index}_Label"

        return {'FINISHED'}

# Operator to select a component by its part name
class MICROBI_OT_select_component(bpy.types.Operator):
    bl_idname = "object.microbi_select_component"
    bl_label = "Select Component"
    bl_description = "Select a component by its name"

    def execute(self, context):
        component_name = context.scene.microbi_component_name
        obj = bpy.data.objects.get(component_name)

        if obj:
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report({'INFO'}, f"Selected component: {component_name}")
        else:
            self.report({'ERROR'}, f"Component '{component_name}' does not exist")

        return {'FINISHED'}

# User interface panel for the Microbi Assembly Sequencer add-on in the 3D view's Tool tab
class MicrobiAssemblySequencerPanel(bpy.types.Panel):
    bl_label = "Microbi Assembly Sequencer"
    bl_idname = "OBJECT_PT_microbi_assembly_sequencer"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Assembly Sequencer'

    def draw(self, context):
        layout = self.layout

        layout.label(text="Prepare Base:")
        row = layout.row()
        row.operator("object.microbi_set_seam", text="Set Seam (Edit Mode)", icon='MESH_UVSPHERE')

        row = layout.row()
        row.operator("object.microbi_clear_seam", text="Clear Seam (Edit Mode)", icon='LOOP_BACK')

        row = layout.row()
        row.operator("object.microbi_uv_unwrap", text="UV Unwrap", icon='MESH_GRID')
        row.operator("object.microbi_rotate_90", text="Rotate 90°", icon='FILE_REFRESH')

        row = layout.row()
        row.operator("object.microbi_create_mst", text="Sort Faces", icon='NODETREE')

        row = layout.row()
        row.operator("object.microbi_toggle_gpencil_visibility", text="Show/Hide MST", icon='HIDE_OFF')

        layout.label(text="Prepare Components:")
        row = layout.row()
        row.operator("object.microbi_create_loose_parts", text="Create Loose Parts", icon="XRAY")

        row = layout.row()
        row.operator("object.microbi_sort_components", text="Sort Components", icon="NODETREE")

        layout.label(text="Utils:")

        row = layout.row()
        row.operator("object.microbi_edge_naming", text="Edge Naming", icon='FONT_DATA')

        layout.prop(context.scene, 'microbi_face_text_size', text="Face Text Size")
        layout.prop(context.scene, 'microbi_edge_text_size', text="Edge Text Size")

        layout.label(text="Select Component:")
        row = layout.row()
        row.prop(context.scene, 'microbi_component_name', text="")
        row.operator("object.microbi_select_component", text="Select", icon='VIEWZOOM')