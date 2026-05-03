from sre_parse import CATEGORIES
from tabnanny import check
from typing import List
from myapp.models import catagory, con, fback, product as prod, Order, recipie, totalorder,sup as Sup, elections, candidate, votes 
from telnetlib import LOGOUT
from django.http import HttpResponse
from django.shortcuts import redirect, render
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.views.decorators.csrf import csrf_exempt
from paytm import checksum as Checksum
from web3 import Web3

# Connect to Ganache
w3 = Web3(Web3.HTTPProvider('http://localhost:7545'))

MERCHANT_KEY='aSDwy39C70SHy8od'


# Create your views here.


def contact(request):
    if request.method == "POST":

        name = request.POST['name']
        email = request.POST['email']

        desc = request.POST['desc']

        ins = con(name=name, email=email, desc=desc)
        ins.save()
        return render(request,'food.html')

    return render(request, 'contact.html')


def food(request):
    return render(request, 'food.html')


def about1(request):
    return render(request, 'about1.html')


def feed(request):
    if request.method == "POST":
        print('this is post')
        des = request.POST['des']
        f = fback(des=des)
        f.save()
        print("saved")
        return render(request,'food.html')
    return render(request, 'feed.html')


def signup(request):
    if request.method == "POST":
        username = request.POST['username']
        if User.objects.filter(username=username).exists():
            messages.info(request, 'Username already exixts!')
            return redirect('signup')
        fname = request.POST['fname']
        lname = request.POST['lname']
        email = request.POST['email']
        phone = request.POST['phone']
        address = request.POST['address']
        pass1 = request.POST['pass1']
        pass2 = request.POST['pass2']
        myuser = User.objects.create_user(username, email, pass1)
        myuser.first_name = fname
        myuser.last_name = lname
        myuser.phone = phone
        ouruser=Sup(username=username,email=email,phone=phone,address=address)
        ouruser.save()
        myuser.save()
        messages.success(request, "Your acoount has been successfully created")
        return redirect('signin')
    return render(request, 'signup.html')


def signin(request):
    if request.method == 'POST':
        username = request.POST['username']
        pass1 = request.POST['pass1']
        user = authenticate(username=username, password=pass1)
        if user is not None:
            login(request, user)
            request.session['userid'] = user.id
            request.session['username'] = user.username
            return render(request, "food.html", {"user": user})
        else:
            messages.error(request, "Bad Credentials!")
    return render(request, 'loginn.html')


def signout(request):
    logout(request)
    return redirect('signin')

def profile(request):
    if request.method=="POST":
        name=request.POST['name']
        address=request.POST['address']
        email=request.POST['email']
        phone=request.POST['phone']
        print(name,address,email,phone)
    name = request.user.username
    sups = Sup.objects.filter(username=name)
    print(name)
    print(sups)
    return render(request,'profile.html',{'sups':sups})


def order(request):
    # Get all elections
    all_elections = elections.objects.all()
    election_id = request.GET.get('ele')
    
    # Check if the user is authenticated
    if request.user.is_authenticated:
        # Get the user's sup instance
        user_sup = Sup.objects.get(username=request.user.username)
        # Get the elections the user has voted for
        voted_elections = votes.objects.filter(user=user_sup).values_list('election', flat=True)
        # Filter out the elections the user has not voted for
        elections_not_voted = all_elections.exclude(id__in=voted_elections)
    else:
        # If the user is not authenticated, show all elections
        elections_not_voted = all_elections
    
    if election_id:
        # Filter candidates based on the specified election
        candidates = candidate.objects.filter(elections_id=election_id)
    else:
        candidates = None

    context = {'elections': elections_not_voted, 'candidates': candidates}
    return render(request, 'order.html',context)


def cart(request):

    ids = list(request.session.get('cart').keys())
    products = prod.objects.filter(id__in=ids)
    print(products)
    return render(request, 'cart.html', {'products': products})


def checkout(request):
    if request.method == 'POST':
        print(request.POST)
        return redirect('cart')


def detail(request):
    amount = (request.GET.get('total'))
    if request.method == 'POST':
        address = request.POST['address']
        phone = request.POST['phone']
        customer = request.session.get('userid')
        name = request.POST['name']
        cart = request.session.get('cart')
        products = prod.objects.filter(id__in=list(cart.keys()))
        am = request.POST['amount']
        

        det = totalorder(name=name, address=address, phone=phone, totalamount=am)
        det.save()
        id=det.orderid

        for product in products:
            ord = Order(orderid=id, product=product, customer_id=customer,quantity=cart.get(str(product.id)), price=product.price)
            ord.save()
        request.session['cart'] = {}
        am=float(am)
       
        param_dict = {

            'MID': 'hUEGRx07177105873953',
            'ORDER_ID': str(id),
            'TXN_AMOUNT': str(am),
            'CUST_ID': str(customer),
            'INDUSTRY_TYPE_ID': 'Retail',
            'WEBSITE': 'WEBSTAGING',
            'CHANNEL_ID': 'WEB',
            'CALLBACK_URL': 'http://127.0.0.1:8000/handlerequest/',
        }
        param_dict['CHECKSUMHASH']=Checksum.generate_checksum(param_dict,MERCHANT_KEY)

        return render(request,'paytm.html',{'param_dict':param_dict})
        
       
    cart = request.session.get('cart')
    print(cart)
    if cart == {}:
        messages.error(request, "PLEASE ADD ITEMS INTO CART...")
        return redirect('cart')
    return render(request, 'details.html', {'amount': amount})


def ourorder(request):
    id = request.GET.get('id')
    myord = Order.objects.filter(orderid=id).order_by('-date')
    ord = totalorder.objects.filter(orderid=id)
    print(myord)
    print(id)
    print(ord)
    return render(request, 'ourorder.html', {'orders': myord,'ord': ord})


def myorder(request):
    name = request.user.username
    torders = totalorder.objects.filter(name=name)
    print(torders)
    return render(request, 'myorder.html', {'morders': torders})


def changepassword(request):
    if request.method == 'POST':
        name = request.session.get('username')
        u = User.objects.get(username=name)
        cpassword = request.POST['oldp']
        password = request.POST['newp']
        user = authenticate(username=name, password=cpassword)
        if user is not None:
            u.set_password(password)
            u.save()
            return render(request, 'food.html')
        else:
            messages.error(request, "please enter correct paassword")
    return render(request, 'changepw.html')


def orderplaced(request):
    return render(request, 'orderplaced.html')


def recepie(request):
    election = elections.objects.all()
    return render(request, 'electrolist.html', {'elections': election}) 


def recepieview(request):
    if request.method == 'POST':
        election_id = request.POST.get('eid')
        candidates = candidate.objects.filter(elections_id=election_id)
        election = elections.objects.get(id=election_id)
        data = {'candidates': candidates, 'election': election}
        return render(request, 'viewresult.html', data)


def services(request):
    return render(request, 'services.html')


@csrf_exempt
def handlerequest(request):
    form = request.POST
    response_dict = {}
    for i in form.keys():
        response_dict[i] = form[i]
        if i == 'CHECKSUMHASH':
            checksum = form[i]

    verify = Checksum.verify_checksum(response_dict, MERCHANT_KEY, checksum)
    print(response_dict["ORDERID"])
    oid=response_dict["ORDERID"]
    if verify:
        if response_dict['RESPCODE'] == '01':
            print('order successful')
        else:
            print('order was not successful because' + response_dict['RESPMSG'])
            totalorder.objects.filter(orderid=oid).delete()
            Order.objects.filter(orderid=oid).delete()
    return render(request, 'orderplaced.html', {'response': response_dict})

def vote_candidate(request):
    if request.method == 'POST':
        candidate_id = request.POST.get('pid')
        candidate_instance = candidate.objects.get(id=candidate_id)
        logged_in_user = request.user
        if candidate_instance:
            candidate_instance.vote += 1
            candidate_instance.save()

            sup_instance = Sup.objects.get(username=logged_in_user.username, email=logged_in_user.email)
            election_instance = candidate_instance.elections
       
            # Create the vote object
            vote = votes.objects.create(
                election=election_instance,
                candidate=candidate_instance,
                user=sup_instance
            )
            vote.save()
            
            # Send transaction to Ganache
            if w3.is_connected():
                # Assume you have a deployed contract 'VotingContract' with a method 'vote'
                contract_address = '0xE7f4cb0A5563228F70F73D26064fFd953aa2baD6'
                contract_abi = [
                {
                    "inputs": [],
                    "stateMutability": "nonpayable",
                    "type": "constructor"
                },
                {
                    "inputs": [
                        {
                            "internalType": "uint256",
                            "name": "_candidateId",
                            "type": "uint256"
                        },
                        {
                            "internalType": "string",
                            "name": "_electionName",
                            "type": "string"
                        },
                        {
                            "internalType": "string",
                            "name": "_userName",
                            "type": "string"
                        },
                        {
                            "internalType": "string",
                            "name": "_candidateName",
                            "type": "string"
                        }
                    ],
                    "name": "vote",
                    "outputs": [],
                    "stateMutability": "nonpayable",
                    "type": "function"
                },
                {
                    "inputs": [
                        {
                            "internalType": "uint256",
                            "name": "",
                            "type": "uint256"
                        }
                    ],
                    "name": "candidates",
                    "outputs": [
                        {
                            "internalType": "uint256",
                            "name": "id",
                            "type": "uint256"
                        },
                        {
                            "internalType": "string",
                            "name": "name",
                            "type": "string"
                        },
                        {
                            "internalType": "uint256",
                            "name": "votes",
                            "type": "uint256"
                        }
                    ],
                    "stateMutability": "view",
                    "type": "function"
                },
                {
                    "inputs": [],
                    "name": "candidatesCount",
                    "outputs": [
                        {
                            "internalType": "uint256",
                            "name": "",
                            "type": "uint256"
                        }
                    ],
                    "stateMutability": "view",
                    "type": "function"
                },
                {
                    "inputs": [
                        {
                            "internalType": "uint256",
                            "name": "",
                            "type": "uint256"
                        },
                        {
                            "internalType": "address",
                            "name": "",
                            "type": "address"
                        }
                    ],
                    "name": "hasVotedForCandidate",
                    "outputs": [
                        {
                            "internalType": "bool",
                            "name": "",
                            "type": "bool"
                        }
                    ],
                    "stateMutability": "view",
                    "type": "function"
                },
                {
                    "inputs": [
                        {
                            "internalType": "uint256",
                            "name": "",
                            "type": "uint256"
                        },
                        {
                            "internalType": "address",
                            "name": "",
                            "type": "address"
                        }
                    ],
                    "name": "hasVotedInElection",
                    "outputs": [
                        {
                            "internalType": "bool",
                            "name": "",
                            "type": "bool"
                        }
                    ],
                    "stateMutability": "view",
                    "type": "function"
                },
                {
                    "inputs": [],
                    "name": "totalVotes",
                    "outputs": [
                        {
                            "internalType": "uint256",
                            "name": "",
                            "type": "uint256"
                        }
                    ],
                    "stateMutability": "view",
                    "type": "function"
                },
                {
                    "inputs": [
                        {
                            "internalType": "uint256",
                            "name": "",
                            "type": "uint256"
                        },
                        {
                            "internalType": "address",
                            "name": "",
                            "type": "address"
                        }
                    ],
                    "name": "votes",
                    "outputs": [
                        {
                            "internalType": "uint256",
                            "name": "candidateId",
                            "type": "uint256"
                        },
                        {
                            "internalType": "string",
                            "name": "electionName",
                            "type": "string"
                        },
                        {
                            "internalType": "string",
                            "name": "candidateName",
                            "type": "string"
                        },
                        {
                            "internalType": "string",
                            "name": "userName",
                            "type": "string"
                        }
                    ],
                    "stateMutability": "view",
                    "type": "function"
                },
                {
                    "inputs": [
                        {
                            "internalType": "address",
                            "name": "",
                            "type": "address"
                        }
                    ],
                    "name": "votesByUser",
                    "outputs": [
                        {
                            "internalType": "uint256",
                            "name": "",
                            "type": "uint256"
                        }
                    ],
                    "stateMutability": "view",
                    "type": "function"
                }
            ]
                contract = w3.eth.contract(address=contract_address, abi=contract_abi)

                tx_hash = contract.functions.vote(candidate_instance.id,election_instance.name,candidate_instance.candidatename,sup_instance.username).transact({'from': w3.eth.accounts[0]})
                tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
            return redirect('order')
    return redirect('order')

