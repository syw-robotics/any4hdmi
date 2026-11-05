from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import mujoco
import numpy as np

from any4hdmi.scripts.viewer_contacts import ContactHighlights, GROUND_COLOR, SELF_COLOR, ground_contact_bodies, self_collision_bodies


MODEL_XML = '''
<mujoco><worldbody>
  <geom name="floor" type="plane" size="5 5 .1"/>
  <body name="pelvis" pos="0 0 .3"><freejoint/><geom size=".03"/>
    <body name="left_ankle_roll_link" pos="0 .08 -.2">
      <joint/><geom size=".1"/>
      <body name="toe" pos=".1 0 -.09"><geom size=".03" contype="0" conaffinity="0"/></body>
    </body>
    <body name="right_ankle_roll_link" pos="0 -.08 -.2"><joint/><geom size=".1"/></body>
  </body>
  <body name="object" pos=".15 .08 .1"><freejoint/><geom size=".1"/></body>
</worldbody></mujoco>
'''


class ViewerContactsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = mujoco.MjModel.from_xml_string(MODEL_XML)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.left = self.model.body("left_ankle_roll_link").id
        self.right = self.model.body("right_ankle_roll_link").id
        self.toe = self.model.body("toe").id

    def test_ground_contacts_include_visual_toes_and_scope(self) -> None:
        feet = ground_contact_bodies(self.model, self.data, 0, feet_only=True)
        self.assertEqual(feet, {self.left, self.right, self.toe})
        all_links = ground_contact_bodies(self.model, self.data, 0, feet_only=False)
        self.assertIn(self.model.body("object").id, all_links)
        self.data.qpos[2] += .2
        mujoco.mj_forward(self.model, self.data)
        self.assertEqual(ground_contact_bodies(self.model, self.data, 0, feet_only=True), set())

    def test_near_ground_feet_with_different_heights_both_highlight(self) -> None:
        self.data.qpos[2] += .001
        self.model.body_pos[self.right, 2] += .00055
        mujoco.mj_forward(self.model, self.data)
        feet = ground_contact_bodies(self.model, self.data, 0, feet_only=True)
        self.assertTrue({self.left, self.right}.issubset(feet))
        self.model.body_pos[self.right, 2] += .004
        mujoco.mj_forward(self.model, self.data)
        feet = ground_contact_bodies(self.model, self.data, 0, feet_only=True)
        self.assertIn(self.left, feet)
        self.assertNotIn(self.right, feet)

    def test_float32_roundoff_at_five_mm_does_not_toggle_contact(self) -> None:
        self_contacts = []
        for offset in (.005 - 3e-8, .005 + 3e-8, .005 - 2e-8):
            self.data.qpos[:] = self.model.qpos0
            self.data.qpos[2] += offset
            mujoco.mj_forward(self.model, self.data)
            self_contacts.append(self.left in ground_contact_bodies(self.model, self.data, 0, feet_only=True))
        self.assertEqual(self_contacts, [True, True, True])
        self.data.qpos[2] += .001  # Real lift-off must still clear the highlight.
        mujoco.mj_forward(self.model, self.data)
        self.assertNotIn(self.left, ground_contact_bodies(self.model, self.data, 0, feet_only=True))

    def test_self_collision_ignores_floor_objects_and_respects_exclusions(self) -> None:
        self.assertEqual(self_collision_bodies(self.model, self.data), {self.left, self.right})
        xml = MODEL_XML.replace("</mujoco>", '<contact><exclude body1="left_ankle_roll_link" body2="right_ankle_roll_link"/></contact></mujoco>')
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        self.assertGreater(data.ncon, 0)  # Floor/object contacts still exist.
        self.assertEqual(self_collision_bodies(model, data), set())

    def test_persistent_contact_has_no_hidden_frame_between_updates(self) -> None:
        from viser import _messages
        from viser._scene_handles import MeshHandle

        server = mock.MagicMock()
        for method in ("add_checkbox", "add_dropdown"):
            getattr(server.gui, method).side_effect = lambda *args, **kwargs: mock.Mock(value=kwargs.get("initial_value"))
        # Use the real Viser visibility setter to inspect outgoing render messages.
        api = mock.Mock()
        handle = object.__new__(MeshHandle)
        object.__setattr__(handle, "_impl", SimpleNamespace(
            api=api, name="/contact_links/test", visible=True, removed=False, props=SimpleNamespace(),
            position=np.zeros(3), wxyz=np.array([1., 0, 0, 0]),
        ))
        server.scene.add_mesh_simple.return_value = handle
        scene = SimpleNamespace(geom_groups_visible=[True, False, True, False, False, False], _mesh_groups=[], _fixed_geom_handles={}, _scene_offset=np.zeros(3))
        highlights = ContactHighlights(server, self.model, self.data, scene)
        highlights.load_motion(np.tile(self.model.qpos0, (10, 1)), .02)
        highlights.ground.value = True
        highlights.update(force=True)
        api._websock_interface.queue_message.reset_mock()
        highlights.update(force=True)
        hidden = [call.args[0] for call in api._websock_interface.queue_message.call_args_list
                  if isinstance(call.args[0], _messages.SetSceneNodeVisibilityMessage) and not call.args[0].visible]
        self.assertEqual(hidden, [], "A continuously touching link must never receive a hide message")

    def test_highlights_priority_pause_layers_and_restore_batched_links(self) -> None:
        server = mock.MagicMock()
        for method in ("add_checkbox", "add_dropdown"):
            getattr(server.gui, method).side_effect = lambda *args, **kwargs: mock.Mock(value=kwargs.get("initial_value"))
        server.scene.add_mesh_simple.side_effect = lambda *args, **kwargs: mock.Mock(**kwargs)
        batch = SimpleNamespace(body_ids=np.array([self.left, self.right, self.toe]), group_id=0, handle=mock.Mock())
        scene = SimpleNamespace(geom_groups_visible=[True, False, True, False, False, False], _mesh_groups=[batch], _fixed_geom_handles={}, _scene_offset=np.array([1, 2, 3]))
        highlights = ContactHighlights(server, self.model, self.data, scene)
        highlights.load_motion(np.tile(self.model.qpos0, (10, 1)), .02)
        highlights.update(force=True)
        self.assertFalse(highlights.handles)
        highlights.ground.value = True
        highlights.ground.on_update.call_args.args[0](None)
        highlights.update()  # No frame advance required.
        self.assertEqual(highlights.handles[self.left, 0].color, GROUND_COLOR)
        np.testing.assert_array_equal(batch.handle.batched_scales, [0, 0, 0])
        highlights.self_collision.value = True
        highlights.self_collision.on_update.call_args.args[0](None)
        highlights.update()
        self.assertEqual(highlights.handles[self.left, 0].color, SELF_COLOR)
        self.assertEqual(highlights.handles[self.right, 0].color, SELF_COLOR)
        self.assertEqual(highlights.handles[self.toe, 0].color, GROUND_COLOR)
        scene.geom_groups_visible[0] = False
        highlights.update()
        self.assertTrue(all(not h.visible for h in highlights.handles.values()))
        scene.geom_groups_visible[0] = True
        highlights.update()
        self.assertTrue(highlights.handles[self.left, 0].visible)
        highlights.ground.value = highlights.self_collision.value = False
        highlights.ground.on_update.call_args.args[0](None)
        highlights.update()
        np.testing.assert_array_equal(batch.handle.batched_scales, [1, 1, 1])
        self.assertTrue(all(not h.visible for h in highlights.handles.values()))
        # A shared batch must retain the unaffected right foot's original mesh.
        self.model.body_pos[self.right, 2] += .2
        mujoco.mj_forward(self.model, self.data)
        highlights.load_motion(np.tile(self.model.qpos0, (10, 1)), .02)
        highlights.ground.value = True
        highlights.update(force=True)
        np.testing.assert_array_equal(batch.handle.batched_scales, [0, 1, 0])
        np.testing.assert_array_equal(highlights.handles[self.left, 0].position, self.data.xpos[self.left] + scene._scene_offset)
        scene._scene_offset = np.zeros(3)
        highlights.update()  # Camera tracking can change while paused.
        np.testing.assert_array_equal(highlights.handles[self.left, 0].position, self.data.xpos[self.left])

    def test_support_rendering_is_seek_independent_and_cleared_on_clip_replacement(self) -> None:
        server = mock.MagicMock()
        for method in ("add_checkbox", "add_dropdown"):
            getattr(server.gui, method).side_effect = lambda *args, **kwargs: mock.Mock(value=kwargs.get("initial_value"))
        server.scene.add_mesh_simple.side_effect = lambda *args, **kwargs: mock.Mock(**kwargs)
        scene = SimpleNamespace(geom_groups_visible=[True, False, True, False, False, False],
                                _mesh_groups=[], _fixed_geom_handles={}, _scene_offset=np.zeros(3))
        highlights = ContactHighlights(server, self.model, self.data, scene)
        highlights.ground.value = True
        highlights.update(force=True)
        self.assertFalse(highlights.handles)  # A pose alone cannot prove support.
        qpos = np.tile(self.model.qpos0, (30, 1))
        qpos[15:, 0] += np.arange(15) * .01
        highlights.load_motion(qpos, .02)
        for frame, expected in [(5, True), (25, False), (5, True), (25, False)]:
            self.data.qpos[:] = qpos[frame]
            mujoco.mj_forward(self.model, self.data)
            highlights.update(frame=frame)
            self.assertEqual(highlights.handles[self.left, 0].visible, expected)
            self.assertEqual(highlights.handles[self.toe, 0].visible, expected)
        highlights.load_motion(qpos[:1], .02)
        highlights.update(frame=0)
        self.assertFalse(any(handle.visible for handle in highlights.handles.values()))


if __name__ == "__main__":
    unittest.main()
