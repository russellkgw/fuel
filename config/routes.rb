# For details on the DSL available within this file, see http://guides.rubyonrails.org/routing.html

Rails.application.routes.draw do

  root 'session#index'
  post 'session' => 'session#create'

  get 'data_import' => 'data_import#index'
  post 'data_import' => 'data_import#create'

  # get 'data_api' => 'data_api#create_exchange'
  post 'data_api' => 'data_api#create_exchange'
end
