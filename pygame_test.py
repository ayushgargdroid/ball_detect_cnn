import pygame
pygame.init()
screen = pygame.display.set_mode([400,500])
screen.fill([255,255,255])
pygame.font.init()
myfont = pygame.font.SysFont(None, 25)
done = False
val = ['a','b','c','d']
while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
            break
        if event.type == pygame.KEYDOWN:
            print 'sdfsdf'
            if event.key == pygame.K_w:
                print 'yolo'
    for i in range(len(val)):
        fontsurf = myfont.render(val[i], False, [0, 0, 0])
        screen.blit(fontsurf,(0,i*18))
    pygame.display.update()
    # print(pygame.key.get_pressed())
    # for i in pygame.key.get_pressed():
    #     if i is not 0:
    #         print 'asdas'
    # if(pygame.key.get_pressed()[pygame.K_w]!=0):
    #     print 'up'
    # elif(pygame.key.get_pressed()[pygame.K_s]!=0):
    #     print 'down'