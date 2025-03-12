from django.shortcuts import render,redirect
from django.http import HttpResponse
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .models import Project
from .forms import ProjectForm, ReviewForm

# Create your views here.

def projects(request):
    projectObjs = Project.objects.all()
    return render(request, 'projects/projects.html', {'projList' : projectObjs})

def singleproject(request, pk):
    projectObj = Project.objects.get(id=pk)
    form = ReviewForm()

    if request.method == 'POST':
        form = ReviewForm(request.POST)
        review = form.save(commit=False)
        review.project = projectObj
        review.owner = request.user.profile
        review.save()


        projectObj.getVotecount

        messages.success(request, 'Your review was successfully submitted!')
        return redirect('project', pk=projectObj.id)


    return render(request, 'projects/single-project.html',{ 'projectObj' : projectObj, 'form':form })

@login_required(login_url="login")
def createproject(request):
    
    profile = request.user.profile
    form = ProjectForm()
    if (request.method == 'POST'):
        form = ProjectForm(request.POST, request.FILES)
        if form.is_valid():
            project = form.save(commit=False)
            project.owner = profile
            project.save()
            return redirect('account')
    
    context = { 'form' : form }
    return render(request, 'projects/projectform.html', context)

@login_required(login_url="login")
def updateproject(request, pk):

    profile = request.user.profile
    project = profile.project_set.get(id=pk)
    form = ProjectForm(instance=project)
    if (request.method == 'POST'):
        form = ProjectForm(request.POST, request.FILES, instance=project)
        if form.is_valid():
            form.save()
            return redirect('account')
    
    context = { 'form' : form }
    return render(request, 'projects/projectform.html', context)


@login_required(login_url="login")
def deleteproject(request, pk):

    profile = request.user.profile
    project=profile.project_set.get(id=pk)
    
    if request.method=='POST':
        project.delete()
        return redirect('account')
    context = { 'object' : project.title }
    return render(request, 'delete-template.html', context)