import cv2
import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions
import static_obstacle_task

FORWARD_KEY = "w"
LEFT_KEY = "a"
RIGHT_KEY = "d"
FINISH = "f"


def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]


def example():
    env = habitat.Env(
        config=habitat.get_config("pointnav_static.yaml")
    )

    print("Environment creation successful")
    observations = env.reset()
    print("Destination, distance: {:3f}, theta(radians): {:.2f}".format(
        observations["pointgoal_with_gps_compass"][0],
        observations["pointgoal_with_gps_compass"][1]))
    #birdseye = env.sim.agents[0].get_state()['birdseye']
    #birdseye = env.sim._prev_sim_obs['birdseye']
    cv2.imshow("RGB", transform_rgb_bgr(observations['rgb']))

    print("Agent stepping around inside environment.")

    count_steps = 0
    while not env.episode_over:
        keystroke = cv2.waitKey(0)

        if keystroke == ord(FORWARD_KEY):
            action = HabitatSimActions.MOVE_FORWARD
            print("action: FORWARD")
        elif keystroke == ord(LEFT_KEY):
            action = HabitatSimActions.TURN_LEFT
            print("action: LEFT")
        elif keystroke == ord(RIGHT_KEY):
            action = HabitatSimActions.TURN_RIGHT
            print("action: RIGHT")
        elif keystroke == ord(FINISH):
            action = HabitatSimActions.STOP
            print("action: FINISH")
        else:
            print("INVALID KEY")
            env.close()
            env = habitat.Env(
                config=habitat.get_config("pointnav.yaml")
            )
            env.reset()
            continue

        observations = env.step(action)
        collided = env.sim._prev_sim_obs['collided']
        count_steps += 1

        print("Destination, distance: {:3f}, theta(radians): {:.2f}".format(
            observations["pointgoal_with_gps_compass"][0],
            observations["pointgoal_with_gps_compass"][1]))
        print(f'Collided: {collided}')
        cv2.imshow("RGB", transform_rgb_bgr(observations['rgb']))

    print("Episode finished after {} steps.".format(count_steps))

    if (
            action == HabitatSimActions.STOP
            and observations["pointgoal_with_gps_compass"][0] < 0.2
    ):
        print("you successfully navigated to destination point")
    else:
        print("your navigation was unsuccessful")


if __name__ == "__main__":
    example()
