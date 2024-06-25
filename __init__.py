# This program is free software: you can redistribute it and/or modify 
# it under the terms of the GNU General Public License as published by 
# the Free Software Foundation, either version 2 of the License, or
# at your option) any later version.
#
# This program is distributed in the hope that it will be useful, 
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

bl_info = {
    "name": "Microbi Assembly Sequencer",
    "blender": (3, 6, 7),
    "category": "Object",
    "description": "A toolset for creating an assembly sequence of multi-part designs",
    "author": "Laura Maria Gonzalez",
    "version": (1, 1),
    "location": "View3D > Tool"
}

import bpy
import os
import sys

# Ensure the correct path for imports
addon_dir = os.path.dirname(__file__)
if addon_dir not in sys.path:
    sys.path.append(addon_dir)

print("Initializing Microbi Assembly Sequencer...")

from .addon import (
    update_face_text_size,
    update_edge_text_size,
    update_line_weight,
    FaceDataPropertyGroup,
    SortedFaceIndexPropertyGroup,
    MICROBI_OT_set_seam,
    MICROBI_OT_clear_seam,
    MICROBI_OT_uv_unwrap,
    MICROBI_OT_rotate_90,
    MICROBI_OT_create_mst,
    MICROBI_OT_toggle_gpencil_visibility,
    MICROBI_OT_create_loose_parts,
    MICROBI_OT_sort_components,
    MICROBI_OT_edge_naming,
    MICROBI_OT_select_component,
    MicrobiAssemblySequencerPanel,
)

print("Import successful")

def register():
    bpy.utils.register_class(FaceDataPropertyGroup)
    bpy.utils.register_class(SortedFaceIndexPropertyGroup)
    bpy.types.Scene.face_data_3d = bpy.props.CollectionProperty(type=FaceDataPropertyGroup)
    bpy.types.Scene.face_data_2d = bpy.props.CollectionProperty(type=FaceDataPropertyGroup)
    bpy.types.Scene.sorted_face_indices = bpy.props.CollectionProperty(type=SortedFaceIndexPropertyGroup)

    bpy.types.Scene.microbi_face_text_size = bpy.props.FloatProperty(
        name="Face Text Size",
        description="Size of the text created by the sequencer for faces",
        default=0.01,
        min=0.001,
        max=100,
        update=update_face_text_size
    )
    bpy.types.Scene.microbi_edge_text_size = bpy.props.FloatProperty(
        name="Edge Text Size",
        description="Size of the text created by the sequencer for edges",
        default=0.01,
        min=0.001,
        max=100,
        update=update_edge_text_size
    )
    bpy.types.Scene.microbi_component_name = bpy.props.StringProperty(
        name="Component Name",
        description="Name of the component to select",
    )
    bpy.types.Scene.microbi_line_weight = bpy.props.FloatProperty(
        name="Line Weight",
        description="Thickness of the Grease Pencil lines",
        default=10.0,
        min=1.0,
        max=100.0,
        update=update_line_weight
    )
    bpy.utils.register_class(MicrobiAssemblySequencerPanel)
    bpy.utils.register_class(MICROBI_OT_set_seam)
    bpy.utils.register_class(MICROBI_OT_clear_seam)
    bpy.utils.register_class(MICROBI_OT_uv_unwrap)
    bpy.utils.register_class(MICROBI_OT_rotate_90)
    bpy.utils.register_class(MICROBI_OT_create_mst)
    bpy.utils.register_class(MICROBI_OT_toggle_gpencil_visibility)
    bpy.utils.register_class(MICROBI_OT_create_loose_parts)
    bpy.utils.register_class(MICROBI_OT_sort_components)
    bpy.utils.register_class(MICROBI_OT_edge_naming)
    bpy.utils.register_class(MICROBI_OT_select_component)

def unregister():
    del bpy.types.Scene.face_data_3d
    del bpy.types.Scene.face_data_2d
    del bpy.types.Scene.microbi_face_text_size
    del bpy.types.Scene.microbi_edge_text_size
    del bpy.types.Scene.microbi_component_name
    del bpy.types.Scene.microbi_line_weight

    bpy.utils.unregister_class(MicrobiAssemblySequencerPanel)
    bpy.utils.unregister_class(MICROBI_OT_set_seam)
    bpy.utils.unregister_class(MICROBI_OT_clear_seam)
    bpy.utils.unregister_class(MICROBI_OT_uv_unwrap)
    bpy.utils.unregister_class(MICROBI_OT_rotate_90)
    bpy.utils.unregister_class(MICROBI_OT_create_mst)
    bpy.utils.unregister_class(MICROBI_OT_toggle_gpencil_visibility)
    bpy.utils.unregister_class(MICROBI_OT_create_loose_parts)
    bpy.utils.unregister_class(MICROBI_OT_sort_components)
    bpy.utils.unregister_class(MICROBI_OT_edge_naming)
    bpy.utils.unregister_class(MICROBI_OT_select_component)

if __name__ == "__main__":
    register()
