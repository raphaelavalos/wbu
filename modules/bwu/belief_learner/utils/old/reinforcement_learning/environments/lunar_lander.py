import numpy as np
from gym.envs.box2d import LunarLander
from gym.envs.box2d.lunar_lander import ContactDetector, VIEWPORT_W, SCALE, VIEWPORT_H, INITIAL_RANDOM, LANDER_POLY, \
    LEG_AWAY, LEG_W, LEG_H, LEG_DOWN, LEG_SPRING_TORQUE
from gym.vector.utils import spaces

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

INITIAL_RANDOM = INITIAL_RANDOM


class LunarLanderRandomInit(LunarLander):
    smooth_random = True

    def reset(self):
        self._destroy()
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.game_over = False
        self.prev_shaping = None

        W = VIEWPORT_W / SCALE
        H = VIEWPORT_H / SCALE

        # terrain
        CHUNKS = 11
        height = self.np_random.uniform(0, H / 2, size=(CHUNKS + 1,))
        chunk_x = [W / (CHUNKS - 1) * i for i in range(CHUNKS)]
        self.helipad_x1 = chunk_x[CHUNKS // 2 - 1]
        self.helipad_x2 = chunk_x[CHUNKS // 2 + 1]
        self.helipad_y = H / 4
        height[CHUNKS // 2 - 2] = self.helipad_y
        height[CHUNKS // 2 - 1] = self.helipad_y
        height[CHUNKS // 2 + 0] = self.helipad_y
        height[CHUNKS // 2 + 1] = self.helipad_y
        height[CHUNKS // 2 + 2] = self.helipad_y
        smooth_y = [0.33 * (height[i - 1] + height[i + 0] + height[i + 1]) for i in range(CHUNKS)]

        self.moon = self.world.CreateStaticBody(shapes=edgeShape(vertices=[(0, 0), (W, 0)]))
        self.sky_polys = []
        for i in range(CHUNKS - 1):
            p1 = (chunk_x[i], smooth_y[i])
            p2 = (chunk_x[i + 1], smooth_y[i + 1])
            self.moon.CreateEdgeFixture(
                vertices=[p1, p2],
                density=0,
                friction=0.1)
            self.sky_polys.append([p1, p2, (p2[0], H), (p1[0], H)])

        self.moon.color1 = (0.0, 0.0, 0.0)
        self.moon.color2 = (0.0, 0.0, 0.0)

        if self.smooth_random:
            initial_x = np.clip(self.np_random.normal(loc=W / 2., scale=W / 3.),
                                a_min=0., a_max=W)
            initial_y = np.clip(self.np_random.normal(loc=H - (H - self.helipad_y) / 3., scale=self.helipad_y / 3.),
                                a_min=self.helipad_y, a_max=H)
            initial_angle = self.np_random.normal(loc=0., scale=np.pi / 3.)
        else:
            initial_x = self.np_random.uniform(0., W)
            initial_y = self.np_random.uniform(self.helipad_y, H)
            initial_angle = self.np_random.uniform(-np.pi / 2, np.pi / 2)

        self.lander = self.world.CreateDynamicBody(
            position=(initial_x, initial_y),
            angle=initial_angle,
            fixtures=fixtureDef(
                shape=polygonShape(vertices=[(x / SCALE, y / SCALE) for x, y in LANDER_POLY]),
                density=5.0,
                friction=0.1,
                categoryBits=0x0010,
                maskBits=0x001,  # collide only with ground
                restitution=0.0)  # 0.99 bouncy
        )
        self.lander.color1 = (0.5, 0.4, 0.9)
        self.lander.color2 = (0.3, 0.3, 0.5)
        self.lander.ApplyForceToCenter((
            self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
            self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM)
        ), True)

        self.legs = []
        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                position=(initial_x - i * LEG_AWAY / SCALE, initial_y),
                angle=(i * 0.05 + initial_angle),
                fixtures=fixtureDef(
                    shape=polygonShape(box=(LEG_W / SCALE, LEG_H / SCALE)),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001)
            )
            leg.ground_contact = False
            leg.color1 = (0.5, 0.4, 0.9)
            leg.color2 = (0.3, 0.3, 0.5)
            rjd = revoluteJointDef(
                bodyA=self.lander,
                bodyB=leg,
                localAnchorA=(0, 0),
                localAnchorB=(i * LEG_AWAY / SCALE, LEG_DOWN / SCALE),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=LEG_SPRING_TORQUE,
                motorSpeed=+0.3 * i  # low enough not to jump back into the sky
            )
            if i == -1:
                rjd.lowerAngle = +0.9 - 0.5  # The most esoteric numbers here, angled legs have freedom to travel within
                rjd.upperAngle = +0.9
            else:
                rjd.lowerAngle = -0.9
                rjd.upperAngle = -0.9 + 0.5
            leg.joint = self.world.CreateJoint(rjd)
            self.legs.append(leg)

        self.drawlist = [self.lander] + self.legs

        return self.step(np.array([0, 0]) if self.continuous else 0)[0]


class LunarLanderContinuousRandomInit(LunarLanderRandomInit):
    continuous = True


class LunarLanderNoRewardShaping(LunarLander):

    def step(self, action):
        prev_shaping = self.prev_shaping
        state, reward, done, d = super().step(action)
        shaping = self.prev_shaping
        if prev_shaping is not None:
            reward += prev_shaping - shaping
        return state, reward, done, d


class LunarLanderContinuousNoRewardShaping(LunarLanderNoRewardShaping):
    continuous = True


class LunarLanderRewardShapingAugmented(LunarLander):

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(9,), dtype=np.float32)

    def step(self, action):
        state, reward, done, d = super().step(action)
        state = np.append(state, [self.prev_shaping * 1e-2], axis=-1)
        return state, reward, done, d


class LunarLanderContinuousRewardShapingAugmented(LunarLanderRewardShapingAugmented):
    continuous = True


class LunarLanderRandomInitRewardShapingAugmented(LunarLanderRandomInit, LunarLanderRewardShapingAugmented):
    def reset(self):
        return LunarLanderRandomInit.reset(self)

    def step(self, action):
        return super().step(action)


class LunarLanderRandomInitNoRewardShaping(LunarLanderRandomInit, LunarLanderNoRewardShaping):
    def reset(self):
        return LunarLanderRandomInit.reset(self)

    def step(self, action):
        return super().step(action)


class LunarLanderContinuousRandomInitNoRewardShaping(LunarLanderRandomInitNoRewardShaping):
    continuous = True


class LunarLanderContinuousRandomInitRewardShapingAugmented(LunarLanderRandomInitRewardShapingAugmented):
    continuous = True
