import sys
import random
from itertools import cycle
import argparse

import pygame
from pygame.locals import QUIT, KEYDOWN, K_ESCAPE

from flappy_numpy import init_pool as numpy_init_pool
from flappy_numpy import save_pool as numpy_save_pool
from flappy_numpy import model_crossover as numpy_model_crossover
from flappy_numpy import model_mutate as numpy_model_mutate
from flappy_numpy import change_weights as numpy_change_weights
from flappy_numpy import predict_action as numpy_predict_action

from flappy_torch import init_pool as torch_init_pool
from flappy_torch import save_pool as torch_save_pool
from flappy_torch import model_crossover as torch_model_crossover
from flappy_torch import model_mutate as torch_model_mutate
from flappy_torch import change_weights as torch_change_weights
from flappy_torch import predict_action as torch_predict_action

init_pool = None
save_pool = None
model_crossover = None
model_mutate = None
change_weights = None
predict_action = None



FPS = 30
SCREENWIDTH  = 288.0
SCREENHEIGHT = 512.0
PIPEGAPSIZE  = 100
BASEY        = SCREENHEIGHT * 0.79


def init_backend(backend):
    global init_pool, save_pool, model_crossover, model_mutate, change_weights, predict_action
    if backend == 'numpy':
        init_pool = numpy_init_pool
        save_pool = numpy_save_pool
        model_crossover = numpy_model_crossover
        model_mutate = numpy_model_mutate
        change_weights = numpy_change_weights
        predict_action = numpy_predict_action
    elif backend == 'cupy':
        raise Exception(f'"{backend}" backend not implemented')
    elif backend == 'torch':
        init_pool = torch_init_pool
        save_pool = torch_save_pool
        model_crossover = torch_model_crossover
        model_mutate = torch_model_mutate
        change_weights = torch_change_weights
        predict_action = torch_predict_action
    else:
        raise Exception(f'Unknown backend "{backend}"')
    print(f'Loaded backend "{backend}"')


def load_image(path, alpha=True, rotate=None):
    properties = {}
    if alpha:
        properties['image'] = pygame.image.load(path).convert_alpha()
    else:
        properties['image'] = pygame.image.load(path).convert()
    if rotate is not None:
        properties['image'] = pygame.transform.rotate(properties['image'], rotate)
    properties['height'] = properties['image'].get_height()
    properties['width'] = properties['image'].get_width()
    return properties


def init_pygame():
    env = {}
    env['USE_PYGAME'] = True
    pygame.init()
    env['clock'] = pygame.time.Clock()
    env['screen'] = pygame.display.set_mode((int(SCREENWIDTH), int(SCREENHEIGHT)))
    pygame.display.set_caption('Flappy Bird')

    env['numbers'] = []
    for i in range(10):
        env['numbers'].append(load_image('assets/sprites/' + str(i) + '.png'))

    env['gameover'] = load_image('assets/sprites/gameover.png')
    env['message'] = load_image('assets/sprites/message.png')
    env['base'] = load_image('assets/sprites/base.png')

    # sounds
    if 'win' in sys.platform:
        soundExt = '.wav'
    else:
        soundExt = '.ogg'

    env['sounds'] = {}
    env['sounds']['die']    = pygame.mixer.Sound('assets/audio/die' + soundExt)
    env['sounds']['hit']    = pygame.mixer.Sound('assets/audio/hit' + soundExt)
    env['sounds']['point']  = pygame.mixer.Sound('assets/audio/point' + soundExt)
    env['sounds']['swoosh'] = pygame.mixer.Sound('assets/audio/swoosh' + soundExt)
    env['sounds']['wing']   = pygame.mixer.Sound('assets/audio/wing' + soundExt)

    # While True:
    # There are several backgrounds, but we will use only one
    env['background'] = load_image('assets/sprites/background-day.png', alpha=False)

    bird_imgs = (
        'assets/sprites/bluebird-upflap.png',
        'assets/sprites/bluebird-midflap.png',
        'assets/sprites/bluebird-downflap.png',
    )
    env['player'] = [
        load_image(bird_imgs[i]) for i in range(3)
    ]

    pipe_img = 'assets/sprites/pipe-green.png'
    env['pipe'] = [
        load_image(pipe_img, rotate=180),
        load_image(pipe_img)
    ]

    env['player'][0]
    for i in range(3):
        env['player'][i]['hitmask'] = getHitmask(env['player'][i]['image'])

    for i in range(2):
        env['pipe'][i]['hitmask'] = getHitmask(env['pipe'][i]['image'])

    env['playerY'] = int((SCREENHEIGHT - env['player'][0]['height']) / 2)
    env['baseX'] = 0
    env['playerIndexGen'] = cycle([0, 1, 2, 1])
    return env


def draw_frame(env, pool, score,
               upperPipes, lowerPipes,
               playerIndex, playersXList, playersYList, playersState):
    # draw sprites
    env['screen'].blit(env['background']['image'], (0,0))

    for uPipe, lPipe in zip(upperPipes, lowerPipes):
        env['screen'].blit(env['pipe'][0]['image'], (uPipe['x'], uPipe['y']))
        env['screen'].blit(env['pipe'][1]['image'], (lPipe['x'], lPipe['y']))

    env['screen'].blit(env['base']['image'], (env['baseX'], BASEY))
    # print score so player overlaps the score
    showScore(env, score)
    for idx in range(pool['len']):
        if playersState[idx]:
            env['screen'].blit(env['player'][playerIndex]['image'], (playersXList[idx], playersYList[idx]))

    pygame.display.update()
    env['clock'].tick(FPS)


def mainGame(env, pool):
    global fitness
    score = playerIndex = loopIter = 0
    playerIndexGen = env['playerIndexGen']
    playersXList = []
    playersYList = []
    for idx in range(pool['len']):
        playerX, playerY = int(SCREENWIDTH * 0.2), env['playerY']
        playersXList.append(playerX)
        playersYList.append(playerY)
    baseShift = env['base']['width'] - env['background']['width']

    # get 2 new pipes to add to upperPipes lowerPipes list
    newPipe1 = getRandomPipe(env)
    newPipe2 = getRandomPipe(env)

    # list of upper pipes
    upperPipes = [
        {'x': SCREENWIDTH + 200, 'y': newPipe1[0]['y']},
        {'x': SCREENWIDTH + 200 + (SCREENWIDTH / 2), 'y': newPipe2[0]['y']},
    ]

    # list of lowerpipe
    lowerPipes = [
        {'x': SCREENWIDTH + 200, 'y': newPipe1[1]['y']},
        {'x': SCREENWIDTH + 200 + (SCREENWIDTH / 2), 'y': newPipe2[1]['y']},
    ]

    next_pipe_x = lowerPipes[0]['x']
    next_pipe_hole_y = (lowerPipes[0]['y'] + (upperPipes[0]['y'] + env['pipe'][0]['height']))/2

    pipeVelX = -4

    # player velocity, max velocity, downward accleration, accleration on flap
    playersVelY    =  []   # player's velocity along Y, default same as playerFlapped
    playerMaxVelY =  10   # max vel along Y, max descend speed
    playerMinVelY =  -8   # min vel along Y, max ascend speed
    playersAccY    =  []   # players downward accleration
    playerFlapAcc =  -9   # players speed on flapping
    playersFlapped = [] # True when player flaps
    playersState = []    # True if player is alive

    for idx in range(pool['len']):
        playersVelY.append(-9)
        playersAccY.append(1)
        playersFlapped.append(False)
        playersState.append(True)

    alive_players = pool['len']


    while True:
        preds = predict_action(pool, playersYList, next_pipe_x, next_pipe_hole_y)
        for idx in range(pool['len']):
            if playersYList[idx] < 0 and playersState[idx]:
                alive_players -= 1
                playersState[idx] = False
            if playersState[idx]:
                pool['fitness'][idx] += 1
                # prediction
                if preds[idx]:
                    if playersYList[idx] > -2 * env['player'][0]['height']:
                        playersVelY[idx] = playerFlapAcc
                        playersFlapped[idx] = True
                        #SOUNDS['wing'].play()

                # movement
                if playersVelY[idx] < playerMaxVelY and not playersFlapped[idx]:
                    playersVelY[idx] += playersAccY[idx]
                if playersFlapped[idx]:
                    playersFlapped[idx] = False
                playerHeight = env['player'][playerIndex]['height']
                playersYList[idx] += min(playersVelY[idx], BASEY - playersYList[idx] - playerHeight)

        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()

        # check for crash here, returns status list
        crashTest = checkCrash(
                env, pool,
                {'x': playersXList, 'y': playersYList, 'index': playerIndex},
                upperPipes, lowerPipes
        )

        for idx in range(pool['len']):
            if playersState[idx] and crashTest[idx]:
                alive_players -= 1
                playersState[idx] = False
        if alive_players == 0:
            return {
                'y': playerY,
                'groundCrash': crashTest[0],
                'basex': env['baseX'],
                'upperPipes': upperPipes,
                'lowerPipes': lowerPipes,
                'score': score,
                'playerVelY': 0,
            }

        # check for score
        for idx in range(pool['len']):
            if playersState[idx]:
                pipe_idx = 0
                playerMidPos = playersXList[idx]
                for pipe in upperPipes:
                    pipeMidPos = pipe['x'] + env['pipe'][0]['width']
                    if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                        next_pipe_x = lowerPipes[pipe_idx+1]['x']
                        next_pipe_hole_y = (lowerPipes[pipe_idx+1]['y'] + (upperPipes[pipe_idx+1]['y'] + env['pipe'][pipe_idx+1]['height'])) / 2
                        score += 1
                        pool['fitness'][idx] += 25
                        # SOUNDS['point'].play()
                    pipe_idx += 1

        # playerIndex basex change
        if (loopIter + 1) % 3 == 0:
            playerIndex = next(playerIndexGen)
        loopIter = (loopIter + 1) % 30
        env['baseX'] = -((-env['baseX'] + 100) % baseShift)

        # move pipes to left
        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            uPipe['x'] += pipeVelX
            lPipe['x'] += pipeVelX

        # add new pipe when first pipe is about to touch left of screen
        if 0 < upperPipes[0]['x'] < 5:
            newPipe = getRandomPipe(env)
            upperPipes.append(newPipe[0])
            lowerPipes.append(newPipe[1])

        # remove first pipe if its out of the screen
        if upperPipes[0]['x'] < -env['pipe'][0]['width']:
            upperPipes.pop(0)
            lowerPipes.pop(0)

        draw_frame(env, pool, score,
                   upperPipes, lowerPipes,
                   playerIndex, playersXList, playersYList, playersState)


def finalize(pool):
    """Perform genetic updates here"""
    new_weights = []
    total_fitness = 0
    fitness = []
    normalized_fitness = []
    for idx in range(pool['len']):
        fitness.append(pool['fitness'][idx])
        total_fitness += fitness[idx]
    for idx in range(pool['len']):
        fitness[idx] /= total_fitness
        normalized_fitness.append(fitness[idx])
        if idx > 0:
            normalized_fitness[idx] += normalized_fitness[idx - 1]
    for _ in range(pool['len'] // 2):
        parent1 = random.uniform(0, 1)
        parent2 = random.uniform(0, 1)
        idx1 = -1
        idx2 = -1
        for idxx in range(pool['len']):
            if fitness[idxx] >= parent1:
                idx1 = idxx
                break
        for idxx in range(pool['len']):
            if fitness[idxx] >= parent2:
                idx2 = idxx
                break
        new_weights1 = model_crossover(pool, idx1, idx2)
        updated_weights1 = model_mutate(new_weights1[0])
        updated_weights2 = model_mutate(new_weights1[1])
        new_weights.append(updated_weights1)
        new_weights.append(updated_weights2)
    for idx in range(len(new_weights)):
        pool['fitness'][idx] = -100
    change_weights(pool, new_weights)


def parse_args():
    parser = argparse.ArgumentParser(description='Script to run flappy bird')
    parser.add_argument('-n', '--n_loops', type=int,
                        default=1,
                        help='number of training loops')

    parser.add_argument('-b', '--backend', type=str,
                        default='numpy',
                        choices=['numpy', 'cupy', 'torch'],
                        help='what backend to use CPU/GPU')

    return parser.parse_args()

def main():
    args = parse_args()
    init_backend(args.backend)
    n = args.n_loops
    for _ in range(n):
        env = init_pygame()
        pool = init_pool()
        env['SAVE_POOL'] = True

        mainGame(env, pool)
        finalize(pool)
        if env['SAVE_POOL']:
            save_pool(pool)


def getRandomPipe(env):
    """returns a randomly generated pipe"""
    # y of gap between upper and lower pipe
    gapY = random.randrange(0, int(BASEY * 0.6 - PIPEGAPSIZE))
    gapY += int(BASEY * 0.2)
    pipeHeight = env['pipe'][0]['height']
    pipeX = SCREENWIDTH + 10

    return [
        {'x': pipeX, 'y': gapY - pipeHeight},  # upper pipe
        {'x': pipeX, 'y': gapY + PIPEGAPSIZE}, # lower pipe
    ]


def showScore(env, score):
    """displays score in center of screen"""
    if not env['USE_PYGAME']:
        return

    scoreDigits = [int(x) for x in list(str(score))]
    totalWidth = 0 # total width of all numbers to be printed

    for digit in scoreDigits:
        totalWidth += env['numbers'][digit]['width']

    Xoffset = (SCREENWIDTH - totalWidth) / 2

    for digit in scoreDigits:
        env['screen'].blit(env['numbers'][digit]['image'], (Xoffset, SCREENHEIGHT * 0.1))
        Xoffset += env['numbers'][digit]['width']


def checkCrash(env, pool, players, upperPipes, lowerPipes):
    """returns True if player collders with base or pipes."""
    statuses = []
    for idx in range(pool['len']):
        statuses.append(False)

    for idx in range(pool['len']):
        statuses[idx] = False
        pi = players['index']
        players['w'] = env['player'][0]['width']
        players['h'] = env['player'][0]['height']
        # if player crashes into ground
        if players['y'][idx] + players['h'] >= BASEY - 1:
            statuses[idx] = True
        playerRect = pygame.Rect(players['x'][idx], players['y'][idx],
                      players['w'], players['h'])
        pipeW = env['pipe'][0]['width']
        pipeH = env['pipe'][0]['height']

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            # upper and lower pipe rects
            uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], pipeW, pipeH)
            lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], pipeW, pipeH)

            # player and upper/lower pipe hitmasks
            pHitMask = env['player'][pi]['hitmask']
            uHitmask = env['pipe'][0]['hitmask']
            lHitmask = env['pipe'][1]['hitmask']

            # if bird collided with upipe or lpipe
            uCollide = pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
            lCollide = pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)

            if uCollide or lCollide:
                statuses[idx] = True
                break
    return statuses

def pixelCollision(rect1, rect2, hitmask1, hitmask2):
    """Checks if two objects collide and not just their rects"""
    rect = rect1.clip(rect2)

    if rect.width == 0 or rect.height == 0:
        return False

    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y

    for x in range(rect.width):
        for y in range(rect.height):
            if hitmask1[x1+x][y1+y] and hitmask2[x2+x][y2+y]:
                return True
    return False

def getHitmask(image):
    """returns a hitmask using an image's alpha."""
    mask = []
    for x in range(image.get_width()):
        mask.append([])
        for y in range(image.get_height()):
            mask[x].append(bool(image.get_at((x,y))[3]))
    return mask

if __name__ == '__main__':
    main()

