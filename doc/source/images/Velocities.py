import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm

def Velocities_1D(n):
    e = 0.2
    fig = plt.figure(0,figsize=(8, 4))
    fig.clf()
    plt.ion()
    plt.hold(True)
    plt.text(0,0,'0',color='red',horizontalalignment='center',verticalalignment='center',fontsize=20)
    for k in xrange(1,n+1):
        plt.plot([k-1+e,k-e],[0,0],'k-')
        plt.plot([-k+1-e,-k+e],[0,0],'k-')
        plt.text(k,0,str(2*k-1),horizontalalignment='center',verticalalignment='center',fontsize=20)
        plt.text(-k,0,str(2*k),horizontalalignment='center',verticalalignment='center',fontsize=20)
    k=n+1
    plt.plot([k-1+e,k-e],[0,0],'k:')
    plt.plot([-k+1-e,-k+e],[0,0],'k:')
    plt.axis('off')
    plt.title("Velocities numbering 1D",fontsize=20)
    plt.draw()
    plt.hold(False)
    plt.ioff()
    plt.savefig('Velocities_1D.jpeg', dpi = 80)
    
def Velocities_2D(n):
    e = 0.2
    fig = plt.figure(0,figsize=(8,8))
    fig.clf()
    plt.ion()
    plt.hold(True)
    plt.text(0,0,'0',color='red',horizontalalignment='center',verticalalignment='center',fontsize=20)
    compt = 0
    for k in xrange(1,n+1):
        plt.plot([k-1+e,k-e],[0,0],'k-')             # x>0, y=0
        plt.plot([k-1+e,k-e],[k-1+e,k-e],'k-')       # x>0, y=x
        plt.plot([0,0],[k-1+e,k-e],'k-')             # x=0, y>0
        plt.plot([-k+e,-k+1-e],[k-e,k-1+e],'k-')     # x<0, y=-x
        plt.plot([-k+e,-k+1-e],[0,0],'k-')           # x<0, y=0
        plt.plot([-k+e,-k+1-e],[-k+e,-k+1-e],'k-')   # x<0, y=x
        plt.plot([0,0],[-k+e,-k+1-e],'k-')           # x=0, y<0
        plt.plot([k-1+e,k-e],[-k+1-e,-k+e],'k-')     # x>0, y=-x
        compt += 1
        plt.text(k,0,str(compt),horizontalalignment='center',verticalalignment='center',fontsize=20)
        compt += 1
        plt.text(0,k,str(compt),horizontalalignment='center',verticalalignment='center',fontsize=20)
        compt += 1
        plt.text(-k,0,str(compt),horizontalalignment='center',verticalalignment='center',fontsize=20)
        compt += 1
        plt.text(0,-k,str(compt),horizontalalignment='center',verticalalignment='center',fontsize=20)        
        for i in xrange(1,k):
            compt += 1
            plt.text(k,i,str(compt),horizontalalignment='center',verticalalignment='center',fontsize=20)
            compt += 1
            plt.text(i,k,str(compt),horizontalalignment='center',verticalalignment='center',fontsize=20)
            compt += 1
            plt.text(-i,k,str(compt),horizontalalignment='center',verticalalignment='center',fontsize=20)
            compt += 1
            plt.text(-k,i,str(compt),horizontalalignment='center',verticalalignment='center',fontsize=20)
            compt += 1
            plt.text(-k,-i,str(compt),horizontalalignment='center',verticalalignment='center',fontsize=20)
            compt += 1
            plt.text(-i,-k,str(compt),horizontalalignment='center',verticalalignment='center',fontsize=20)
            compt += 1
            plt.text(i,-k,str(compt),horizontalalignment='center',verticalalignment='center',fontsize=20)
            compt += 1
            plt.text(k,-i,str(compt),horizontalalignment='center',verticalalignment='center',fontsize=20)
        compt += 1
        plt.text(k,k,str(compt),horizontalalignment='center',verticalalignment='center',fontsize=20)
        compt += 1
        plt.text(-k,k,str(compt),horizontalalignment='center',verticalalignment='center',fontsize=20)
        compt += 1
        plt.text(-k,-k,str(compt),horizontalalignment='center',verticalalignment='center',fontsize=20)
        compt += 1
        plt.text(k,-k,str(compt),horizontalalignment='center',verticalalignment='center',fontsize=20)
    k=n+1
    plt.plot([k-1+e,k-e],[0,0],'k:')
    plt.plot([-k+1-e,-k+e],[0,0],'k:')
    plt.plot([0,0],[k-1+e,k-e],'k:')
    plt.plot([0,0],[-k+1-e,-k+e],'k:')
    plt.plot([k-1+e,k-e],[k-1+e,k-e],'k:')
    plt.plot([-k+1-e,-k+e],[-k+1-e,-k+e],'k:')
    plt.plot([-k+1-e,-k+e],[k-1+e,k-e],'k:')
    plt.plot([k-1+e,k-e],[-k+1-e,-k+e],'k:')
    plt.axis('off')
    plt.title("Velocities numbering 2D",fontsize=20)
    plt.draw()
    plt.hold(False)
    plt.ioff()
    plt.savefig('Velocities_2D.jpeg', dpi = 80)

if __name__ == "__main__":
    Velocities_1D(2)
    Velocities_2D(2)
