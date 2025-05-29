"""
Test script for DDPGAgent with HEBG integration.
"""

import torch
from torch_geometric.data import Batch, Data

from src.graph.structural_graph import BeamAsNodeGraph
from src.reinforcement.rl_agent import DDPGAgent
from src.utils.model_utils import create_structural_model, load_config


def run_ddpg_hebg_test():
    print("--- Starting DDPG with HEBG Integration Test ---")

    # 1. Create a sample StructuralModel and PyG Data
    print("\nStep 1: Creating sample structural model and PyG data...")
    structural_model = None
    config_path = "config.yaml"  # Ensure this is accessible from where the test is run, or use absolute path

    try:
        try:
            config = load_config(config_path)
            structural_model = create_structural_model(config)
        except FileNotFoundError:
            print(f"Warning: {config_path} not found. Creating a dummy config for testing.")
            dummy_config_content = """
structure_parameters:
  length: 12.0
  width: 8.0
  height: 3.5
  bay_spacing: 4.0
  floor_height: 3.5
  main_beam_spacing_x: 4.0
  main_beam_spacing_y: 4.0
  secondary_beam_spacing_x: 2.0
  secondary_beam_spacing_y: 2.0
  brace_spacing: 4.0
  num_floors: 1
  section_indices:
    frame_beam_section_idx: 0
    main_beam_section_idx: 1
    secondary_beam_section_idx: 2
    brace_beam_section_idx: 3
sections:
  - name: "SHS120x6"
    type: "SHS"
    properties: {A: 0.0025, Ix: 0.000005, Iy: 0.000005, Iz: 0.000005, J: 0.000008}
  - name: "H250x125x6x9"
    type: "H"
    properties: {A: 0.0040, Ix: 0.000050, Iy: 0.000010, Iz: 0.000050, J: 0.000015}
  - name: "C150x75x6"
    type: "C"
    properties: {A: 0.0015, Ix: 0.000010, Iy: 0.000002, Iz: 0.000010, J: 0.000005}
  - name: "SHS100x5"
    type: "SHS"
    properties: {A: 0.0019, Ix: 0.000003, Iy: 0.000003, Iz: 0.000003, J: 0.000005}
"""
            with open(config_path, "w") as f:
                f.write(dummy_config_content)
            print(f"Dummy {config_path} created. Please re-run the test script.")
            return  # Exit for user to re-run

        if not structural_model or not structural_model.beams:
            print("Error: Structural model has no beams. Cannot proceed with the test.")
            return

    except Exception as e:
        print(f"Error during structural model creation: {e}")
        print("Ensure 'config.yaml' is present/creatable and utils work.")
        return

    graph_converter = BeamAsNodeGraph(structural_model)
    pyg_data_single = graph_converter.to_pyg_data()

    if pyg_data_single.num_nodes == 0:
        print("Error: Generated PyG data has 0 nodes. DDPG test cannot run.")
        return
    if pyg_data_single.x is None or pyg_data_single.x.shape[0] == 0 or pyg_data_single.x.shape[1] == 0:
        print("Error: PyG data has no node features or zero feature dimension.")
        return

    print(
        f"Successfully created PyG Data with {pyg_data_single.num_nodes} nodes and {pyg_data_single.x.size(1)} node features."
    )

    # 2. Define configurations
    print("\nStep 2: Defining HEBG and DDPG configurations...")
    hebg_in_channels = pyg_data_single.x.size(1)
    hebg_config = {
        "in_channels": hebg_in_channels,
        "hidden_channels": 32,  # Example
        "out_channels": 48,  # Example, this becomes input to DDPG FC layers
        "num_beam_types": 4,
    }
    action_dim = 5  # Example action dimension
    ddpg_hidden_fc = 128  # Hidden units for FC layers in DDPG actor/critic

    print(f"HEBG Config: {hebg_config}")
    print(f"Action Dimension: {action_dim}")
    print(f"DDPG FC Hidden Dimension: {ddpg_hidden_fc}")

    # 3. Instantiate DDPGAgent
    print("\nStep 3: Instantiating DDPGAgent...")
    try:
        agent = DDPGAgent(
            hebg_config=hebg_config,
            action_dim=action_dim,
            hidden_dim_fc=ddpg_hidden_fc,
            actor_lr=1e-5,  # Using smaller LRs for test stability if any training happens
            critic_lr=1e-4,
        )
        print("DDPGAgent instantiated successfully.")
        # print(agent.actor)
        # print(agent.critic)
    except Exception as e:
        print(f"Error instantiating DDPGAgent: {e}")
        import traceback

        traceback.print_exc()
        return

    # 4. Test select_action
    print("\nStep 4: Testing select_action...")
    try:
        # Ensure model is on the correct device if pyg_data_single was moved
        # pyg_data_single.to(device_of_agent)
        action_selected = agent.select_action(pyg_data_single).squeeze(0)
        print(f"Selected action shape: {action_selected.shape}")
        assert action_selected.shape == (
            action_dim,
        ), f"Expected action shape {(action_dim,)}, got {action_selected.shape}"
        print("select_action test successful.")
    except Exception as e:
        print(f"Error during select_action: {e}")
        import traceback

        traceback.print_exc()
        return

    # 5. Test update method
    print("\nStep 5: Testing update method...")
    try:
        # Create a dummy batch for update
        # For simplicity, use the same data for state and next_state in this test
        # In a real scenario, these would come from a replay buffer
        num_dummy_samples = 1  # Number of experiences in the batch

        # Create a list of Data objects
        # data_list_state = [pyg_data_single.clone() for _ in range(num_dummy_samples)]
        # data_list_next_state = [pyg_data_single.clone() for _ in range(num_dummy_samples)]

        # Create Batch objects using PyG's Batch.from_data_list
        # state_batch = Batch.from_data_list(data_list_state)
        # next_state_batch = Batch.from_data_list(data_list_next_state)
        state_batch = pyg_data_single.clone()
        next_state_batch = pyg_data_single.clone()

        action_batch_tensor = torch.rand(num_dummy_samples, action_dim)
        reward_batch_tensor = torch.rand(num_dummy_samples, 1)
        done_batch_tensor = torch.randint(0, 2, (num_dummy_samples, 1), dtype=torch.float)

        dummy_experience_batch = {
            "state": state_batch,
            "action": action_batch_tensor,
            "reward": reward_batch_tensor,
            "next_state": next_state_batch,
            "done": done_batch_tensor,
        }

        print(f"Dummy batch created with {num_dummy_samples} samples.")
        # print(f"State batch type: {type(state_batch)}, Num graphs: {state_batch.num_graphs}")

        agent.update(dummy_experience_batch)
        print("Update method call successful (no crash).")
        # Further checks could involve loss values if accessible or parameter changes.
    except Exception as e:
        print(f"Error during update method: {e}")
        import traceback

        traceback.print_exc()
        return

    print("\n--- DDPG with HEBG Integration Test Finished Successfully ---")


if __name__ == "__main__":
    run_ddpg_hebg_test()
