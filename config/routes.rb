# For details on the DSL available within this file, see http://guides.rubyonrails.org/routing.html

Rails.application.routes.draw do
  # get '/data-import', to: 'data#index'

  namespace :data do
    get '/import', to: 'import#index'
  end
end
