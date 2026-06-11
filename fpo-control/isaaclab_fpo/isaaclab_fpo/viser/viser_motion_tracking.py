#!/usr/bin/env python3
"""
ViserMotionTracking: Extended visualization for motion tracking tasks with ghost robot.

This class extends ViserIsaacLab to add support for visualizing reference motions
alongside actual robot positions for motion tracking tasks.
"""

from __future__ import annotations

import viser
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Dict, Any

from .viser_isaac_lab import ViserIsaacLab


class ViserMotionTracking(ViserIsaacLab):
    """Extended Viser visualization for motion tracking with ghost robot support."""

    def __init__(
        self,
        asset_dir: Path,
        port: int = 8080,
        host: str = "0.0.0.0",
        num_envs: int = 1,
        update_freq: int = 1,
        show_axes: bool = True,
        axes_size: float = 0.5,
        env_spacing: float = 0.0,
        fps: int = 30,
        random_offsets: Optional[np.ndarray] = None,
        ghost_opacity: float = 0.4,
        ghost_color: Optional[tuple] = None,
    ):
        """Initialize ViserMotionTracking visualization.
        
        Args:
            asset_dir: Directory containing extracted assets
            port: Port for Viser web server
            host: Host address for Viser server
            num_envs: Number of environments to visualize
            update_freq: Update every N simulation steps
            show_axes: Whether to show coordinate axes
            axes_size: Size of coordinate axes
            env_spacing: Spacing between environments when visualizing multiple
            fps: Target frame rate for visualization updates
            random_offsets: Optional array of shape (num_envs, 3) with random XY offsets
            ghost_opacity: Opacity for reference/ghost robot (0-1)
            ghost_color: Optional RGB color for ghost robot, if None uses green tint
        """
        # Store ghost-specific parameters before calling parent init
        self.ghost_opacity = ghost_opacity
        self.ghost_color = ghost_color or (0.3, 1.0, 0.3)  # Default green tint
        
        # Ghost robot state
        self.show_ghost = True
        self.ghost_handles: Dict[int, viser.SceneNodeHandle] = {}
        self.motion_command = None
        
        # Initialize parent class
        super().__init__(
            asset_dir=asset_dir,
            port=port,
            host=host,
            num_envs=num_envs,
            update_freq=update_freq,
            show_axes=show_axes,
            axes_size=axes_size,
            env_spacing=env_spacing,
            fps=fps,
            random_offsets=random_offsets,
        )
        
    def _setup_gui(self):
        """Setup GUI controls including ghost robot controls."""
        # Call parent GUI setup first
        super()._setup_gui()
        
        # Add motion tracking specific controls
        with self.server.gui.add_folder("Motion Tracking"):
            self.gui_show_ghost = self.server.gui.add_checkbox(
                "Show Reference Motion",
                initial_value=self.show_ghost,
            )
            
            self.gui_ghost_opacity = self.server.gui.add_slider(
                "Ghost Opacity",
                min=0.0,
                max=1.0,
                step=0.05,
                initial_value=self.ghost_opacity,
            )
            
            # RGB sliders for ghost color
            self.gui_ghost_color_r = self.server.gui.add_slider(
                "Ghost Color R",
                min=0.0,
                max=1.0,
                step=0.05,
                initial_value=self.ghost_color[0],
            )
            self.gui_ghost_color_g = self.server.gui.add_slider(
                "Ghost Color G", 
                min=0.0,
                max=1.0,
                step=0.05,
                initial_value=self.ghost_color[1],
            )
            self.gui_ghost_color_b = self.server.gui.add_slider(
                "Ghost Color B",
                min=0.0,
                max=1.0,
                step=0.05,
                initial_value=self.ghost_color[2],
            )
            
        # Add callbacks
        @self.gui_show_ghost.on_update
        def _on_show_ghost_change(event):
            self.show_ghost = event.target.value
            self._update_ghost_visibility()
            
        @self.gui_ghost_opacity.on_update
        def _on_ghost_opacity_change(event):
            self.ghost_opacity = event.target.value
            self._update_ghost_appearance()
            
        # Color change callbacks
        @self.gui_ghost_color_r.on_update
        @self.gui_ghost_color_g.on_update  
        @self.gui_ghost_color_b.on_update
        def _on_ghost_color_change(event):
            self.ghost_color = (
                self.gui_ghost_color_r.value,
                self.gui_ghost_color_g.value,
                self.gui_ghost_color_b.value,
            )
            self._update_ghost_appearance()
            
    def load_from_env(self, env):
        """Load robot data from Isaac Lab environment.
        
        Also checks if this is a motion tracking task and prepares ghost robot.
        """
        # Call parent load
        super().load_from_env(env)
        
        # Store environment reference (parent class already stores it as self.env)
        # Check if this is a motion tracking task
        self._check_motion_tracking_task(env)
        
        # Create ghost robot meshes if motion tracking
        if self.motion_command is not None:
            self._create_ghost_robot()
            
    def _check_motion_tracking_task(self, env):
        """Check if environment has motion tracking command."""
        try:
            # Check if command manager has motion command
            if hasattr(env, 'command_manager'):
                for cmd_name, cmd_term in env.command_manager._terms.items():
                    # Check if it's a MotionCommand instance
                    if type(cmd_term).__name__ == 'MotionCommand':
                        self.motion_command = cmd_term
                        print(f"✅ Detected motion tracking task with command: {cmd_name}")
                        return
        except Exception as e:
            print(f"ℹ️  No motion tracking detected: {e}")
            
        self.motion_command = None
        
    def _create_ghost_robot(self):
        """Create ghost robot meshes for reference motion visualization."""
        if self.num_envs > 1:
            # Multi-environment mode
            self._create_batched_ghost_meshes()
        else:
            # Single environment mode
            self._create_single_ghost_meshes()
            
    def _create_single_ghost_meshes(self):
        """Create ghost meshes for single environment mode."""
        print(f"🔄 Creating ghost meshes for single environment...")
        
        with self.server.atomic():
            # Iterate through actual robot mesh handles
            for prim_path, handle in self.mesh_handles.items():
                # Only process visual geometry
                if "visual" not in prim_path.lower() or "robot" not in prim_path.lower():
                    continue
                    
                # Extract body name from prim path
                body_name = self._extract_body_name(prim_path)
                if not body_name:
                    continue
                    
                # Find the body index
                body_idx = None
                for idx, robot_body_name in enumerate(self.robot.body_names):
                    clean_robot_body = robot_body_name.split("/")[-1] if "/" in robot_body_name else robot_body_name
                    if clean_robot_body == body_name or body_name in clean_robot_body or clean_robot_body in body_name:
                        body_idx = idx
                        break
                        
                if body_idx is None:
                    continue
                    
                # Get the mesh file for this prim path
                mesh_file = self.prim_to_mesh.get(prim_path)
                if mesh_file and mesh_file in self.loaded_meshes:
                    mesh = self.loaded_meshes[mesh_file]
                    
                    # Handle trimesh.Scene objects
                    if hasattr(mesh, 'to_geometry'):
                        mesh = mesh.to_geometry()
                    
                    # Create ghost mesh with specified color and opacity
                    ghost_handle = self.server.scene.add_mesh_simple(
                        name=f"/ghost_{body_name}",
                        vertices=mesh.vertices,
                        faces=mesh.faces,
                        color=self.ghost_color,
                        opacity=self.ghost_opacity,
                        flat_shading=True,
                        visible=self.show_ghost,
                    )
                    
                    self.ghost_handles[body_idx] = ghost_handle
                    
        print(f"✅ Created {len(self.ghost_handles)} ghost meshes")
        
    def _create_batched_ghost_meshes(self):
        """Create ghost meshes for multi-environment mode."""
        print(f"🔄 Creating batched ghost meshes for {self.num_envs} environments...")
        
        with self.server.atomic():
            # Create batched ghost versions of visual meshes
            for body_idx in self.body_idx_to_visual_handle:
                body_name = self.robot.body_names[body_idx]
                clean_name = body_name.split("/")[-1] if "/" in body_name else body_name
                
                # Find the mesh for this body - search more thoroughly
                mesh = None
                mesh_found = False
                
                # First try to find exact match
                for prim_path, mesh_file in self.prim_to_mesh.items():
                    if clean_name in prim_path and "visual" in prim_path.lower() and "robot" in prim_path.lower():
                        if mesh_file in self.loaded_meshes:
                            mesh = self.loaded_meshes[mesh_file]
                            mesh_found = True
                            break
                
                # If no mesh found, try less strict matching
                if not mesh_found:
                    for prim_path, mesh_file in self.prim_to_mesh.items():
                        if body_name in prim_path and "visual" in prim_path.lower():
                            if mesh_file in self.loaded_meshes:
                                mesh = self.loaded_meshes[mesh_file]
                                mesh_found = True
                                break
                            
                if mesh is not None:
                    # Handle trimesh.Scene objects
                    if hasattr(mesh, 'to_geometry'):
                        mesh = mesh.to_geometry()
                    
                    # Create batched ghost colors
                    ghost_colors = np.tile(self.ghost_color, (self.num_envs, 1))
                    
                    # Create batched ghost mesh
                    ghost_handle = self.server.scene.add_batched_meshes_simple(
                        name=f"/batched_ghost_{clean_name}",
                        vertices=mesh.vertices,
                        faces=mesh.faces,
                        batched_wxyzs=np.tile([1.0, 0.0, 0.0, 0.0], (self.num_envs, 1)),
                        batched_positions=np.zeros((self.num_envs, 3)),
                        batched_colors=ghost_colors,
                        opacity=self.ghost_opacity,
                        flat_shading=True,
                        visible=self.show_ghost,
                    )
                    
                    self.ghost_handles[body_idx] = ghost_handle
                    print(f"   ✅ Created ghost mesh for body {body_idx} '{clean_name}'")
                else:
                    print(f"   ⚠️  No mesh found for body {body_idx} '{clean_name}'")
                    
        print(f"✅ Created {len(self.ghost_handles)} batched ghost meshes")
        
    def _update_ghost_visibility(self):
        """Update visibility of ghost robot."""
        for handle in self.ghost_handles.values():
            handle.visible = self.show_ghost
            
    def _update_ghost_appearance(self):
        """Update appearance (color/opacity) of ghost robot."""
        # For single env mode, we need to recreate meshes to change color/opacity
        if self.num_envs == 1:
            # Remove existing ghost meshes
            for handle in self.ghost_handles.values():
                handle.remove()
            self.ghost_handles.clear()
            
            # Recreate with new settings
            if self.motion_command is not None:
                self._create_single_ghost_meshes()
        else:
            # For batched mode, update colors
            for body_idx, handle in self.ghost_handles.items():
                ghost_colors = np.tile(self.ghost_color, (self.num_envs, 1))
                # Note: Viser doesn't support changing opacity of existing batched meshes
                # Would need to recreate them for opacity changes
                
    def update_from_env(self, env, force: bool = False, rewards: Optional[torch.Tensor] = None, actions: Optional[torch.Tensor] = None):
        """Update visualization including ghost robot for reference motion.
        
        Args:
            env: Isaac Lab environment (unwrapped)
            force: Force update regardless of update frequency
            rewards: Optional tensor of rewards from env.step()
            actions: Optional tensor of actions sent to env.step()
        """
        # Call parent update for actual robot
        super().update_from_env(env, velocity_commands=False, force=force, rewards=rewards, actions=actions)
        
        # Step count is already tracked by parent class
        
        # Update ghost robot if we have motion tracking
        if self.motion_command is not None and self.show_ghost:
            self._update_ghost_robot(env)
            
    def _update_ghost_robot(self, env):
        """Update ghost robot positions from motion command reference."""
        if not self.ghost_handles:
            return
            
        # Debug: Print motion command body names once
        if not hasattr(self, '_debug_printed_bodies'):
            print(f"\n[DEBUG] Motion command body names ({len(self.motion_command.cfg.body_names)}):")
            for i, name in enumerate(self.motion_command.cfg.body_names):
                print(f"  {i}: {name}")
            print(f"\n[DEBUG] Robot body names ({len(self.robot.body_names)}):")
            for i, name in enumerate(self.robot.body_names):
                print(f"  {i}: {name}")
            self._debug_printed_bodies = True
            
        if self.num_envs == 1:
            # Single environment mode
            env_idx = 0
            
            # Get environment origin
            env_origin = np.zeros(3)
            if hasattr(env, 'scene') and hasattr(env.scene, 'env_origins'):
                env_origin = env.scene.env_origins[env_idx].cpu().numpy()
                
            # Update ghost robot with reference positions
            with self.server.atomic():
                for body_idx, ghost_handle in self.ghost_handles.items():
                    # Get robot body name and clean it
                    robot_body_name = self.robot.body_names[body_idx]
                    clean_robot_name = robot_body_name.split("/")[-1] if "/" in robot_body_name else robot_body_name
                    
                    # Find the corresponding index in motion command body list
                    motion_body_idx = None
                    
                    # TEMP: Visualize the exact reference motion
                    org_ref_motion_body_idx = self.motion_command.robot.body_names.index(clean_robot_name)

                    time_step = self.motion_command.time_steps[env_idx]

                    # Get the reference position and orientation for this body
                    ref_pos_w = self.motion_command.motion._body_pos_w[time_step, org_ref_motion_body_idx].cpu().numpy()
                    ref_quat_w = self.motion_command.motion._body_quat_w[time_step, org_ref_motion_body_idx].cpu().numpy()

                    # Convert position to local coordinates
                    local_pos = ref_pos_w - env_origin

                    # Update ghost position and orientation
                    ghost_handle.position = local_pos
                    ghost_handle.wxyz = ref_quat_w

                    # # Try exact match first
                    # if clean_robot_name in self.motion_command.cfg.body_names:
                    #     motion_body_idx = self.motion_command.cfg.body_names.index(clean_robot_name)
                    # else:
                    #     # Try partial matching
                    #     for i, motion_body_name in enumerate(self.motion_command.cfg.body_names):
                    #         # Check various matching strategies
                    #         if (clean_robot_name == motion_body_name or
                    #             clean_robot_name in motion_body_name or
                    #             motion_body_name in clean_robot_name or
                    #             clean_robot_name.replace("_link", "") == motion_body_name or
                    #             motion_body_name.replace("_link", "") == clean_robot_name):
                    #             motion_body_idx = i
                    #             break
                    
                    # if motion_body_idx is not None:
                    #     # Get reference position and orientation for this body
                    #     ref_pos_w = self.motion_command.body_pos_relative_w[env_idx, motion_body_idx].cpu().numpy()
                    #     ref_quat_w = self.motion_command.body_quat_relative_w[env_idx, motion_body_idx].cpu().numpy()
                        
                    #     # Convert position to local coordinates
                    #     local_pos = ref_pos_w - env_origin
                        
                    #     # Update ghost position and orientation
                    #     ghost_handle.position = local_pos
                    #     ghost_handle.wxyz = ref_quat_w
                    # else:
                    #     # Debug: Report unmatched bodies
                    #     if self.step_count % 100 == 0:
                    #         print(f"[WARNING] No motion match for robot body {body_idx}: '{clean_robot_name}'")
                        
        else:
            # Multi-environment mode
            # Get environment origins
            env_origins = np.zeros((self.num_envs, 3))
            if hasattr(env, 'scene') and hasattr(env.scene, 'env_origins'):
                env_origins = env.scene.env_origins[:self.num_envs].cpu().numpy()
                
            with self.server.atomic():
                for body_idx, ghost_handle in self.ghost_handles.items():
                    # Get robot body name and clean it
                    robot_body_name = self.robot.body_names[body_idx]
                    clean_robot_name = robot_body_name.split("/")[-1] if "/" in robot_body_name else robot_body_name
                    
                    # Find the corresponding index in motion command body list
                    motion_body_idx = None
                    
                    # Try exact match first
                    if clean_robot_name in self.motion_command.cfg.body_names:
                        motion_body_idx = self.motion_command.cfg.body_names.index(clean_robot_name)
                    else:
                        # Try partial matching
                        for i, motion_body_name in enumerate(self.motion_command.cfg.body_names):
                            if (clean_robot_name == motion_body_name or
                                clean_robot_name in motion_body_name or
                                motion_body_name in clean_robot_name or
                                clean_robot_name.replace("_link", "") == motion_body_name or
                                motion_body_name.replace("_link", "") == clean_robot_name):
                                motion_body_idx = i
                                break
                    
                    if motion_body_idx is not None:
                        # Get batched positions and orientations
                        ref_positions = self.motion_command.body_pos_relative_w[:self.num_envs, motion_body_idx].cpu().numpy()
                        ref_quats = self.motion_command.body_quat_relative_w[:self.num_envs, motion_body_idx].cpu().numpy()
                        
                        # Convert to local coordinates and add grid offsets
                        batched_positions = ref_positions - env_origins + self.grid_offsets
                        
                        # Update batched ghost positions
                        ghost_handle.batched_positions = batched_positions
                        ghost_handle.batched_wxyzs = ref_quats